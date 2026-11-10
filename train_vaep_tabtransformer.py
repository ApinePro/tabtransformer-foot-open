"""
train_vaep_tabtransformer.py

使用预训练 TabTransformer encoder（冻结）提取 embedding，
然后在其上训练一个小 MLP 做 VAEP score/concede 双任务预测。

与 train_vaep_baseline.py 的区别：
  - baseline：自训练 cat embedding + MLP
  - this：冻结的预训练 TabTransformer encoder + trainable MLP

为避免提前提取全量 embedding 占用 ~20 GB 内存，
encoder 以 torch.no_grad() 方式在每个 batch 实时运行，
只占用单个 batch 的内存。

动作数据本身缓存在 .vaep_cache/（与 baseline 共享），
训练好的 MLP checkpoint 保存到 checkpoints/vaep_tabtransformer_best.pt。

用法：
  python train_vaep_tabtransformer.py
  python train_vaep_tabtransformer.py --eval-only
  python train_vaep_tabtransformer.py --eval-only --prediction-splits val,test
"""

import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    TabEventEncoder,
    set_seed,
)
from train_vaep_baseline import (
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    SEED,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    VAEPTabularDataset,
    add_vaep_columns,
    binary_log_loss,
    choose_prediction_output_path,
    compute_pos_weight,
    ensure_non_empty_split,
    find_best_threshold,
    load_or_build_actions,
    load_or_build_split_actions,
    safe_num,
    safe_text,
    save_predictions_compact,
    split_train_val_by_match,
    summarize_split,
)

# ---------------------------------------------------------------------------
PRETRAIN_CHECKPOINT = os.path.join("checkpoints", "tabtransformer_row_mask_pretrain.pt")
CHECKPOINT_PATH = os.path.join("checkpoints", "vaep_tabtransformer_best.pt")
PREDICTIONS_OUTPUT = "vaep_tabtransformer_predictions"
METRICS_OUTPUT = "vaep_tabtransformer_metrics.csv"

HIDDEN_DIM = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512          # smaller than baseline because encoder also runs per batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAEP with frozen TabTransformer encoder")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    p.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--prediction-splits", default="val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches", type=int, default=None)
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def resolve_checkpoint_path(path: str) -> str:
    if not os.path.exists(path):
        candidate = os.path.join("checkpoints", os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path


# ---------------------------------------------------------------------------
# Pretrained encoder loading
# ---------------------------------------------------------------------------
def load_pretrained_encoder(ckpt_path: str) -> Tuple[TabEventEncoder, List[str], List[str], dict, List[int], np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cat_cols = list(ckpt["cat_cols"])
    cont_cols = list(ckpt["cont_cols"])
    cat_maps = ckpt["cat_maps"]
    cat_dims = list(ckpt["cat_dims"])
    cont_mean_std = np.asarray(ckpt["cont_mean_std"], dtype=np.float32)

    encoder = TabEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Freeze all encoder parameters
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    return encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std


# ---------------------------------------------------------------------------
# Model: frozen encoder + trainable MLP heads
# ---------------------------------------------------------------------------
class FrozenEncoderVAEP(nn.Module):
    def __init__(self, encoder: TabEventEncoder, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = encoder   # already frozen
        embed_dim = encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.concede_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            emb = self.encoder(x_categ, x_cont)
        h = self.mlp(emb)
        return self.score_head(h).squeeze(-1), self.concede_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / predict helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: FrozenEncoderVAEP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    model.encoder.eval()   # keep encoder in eval mode (BN/dropout)
    total_loss = 0.0
    for x_categ, x_cont, score_lbl, concede_lbl, *_ in tqdm(loader, desc="Train", leave=False):
        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)
        score_lbl = score_lbl.to(device)
        concede_lbl = concede_lbl.to(device)
        optimizer.zero_grad()
        score_logit, concede_logit = model(x_categ, x_cont)
        loss = score_criterion(score_logit, score_lbl) + concede_criterion(concede_logit, concede_lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def predict_dataset(
    model: FrozenEncoderVAEP,
    loader: DataLoader,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
    desc: str = "Predict",
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows = []
    with torch.no_grad():
        for x_categ, x_cont, score_lbl, concede_lbl, match_ids, source_names, action_indices in tqdm(loader, desc=desc, leave=False):
            x_categ = x_categ.to(device)
            x_cont = x_cont.to(device)
            score_lbl_dev = score_lbl.to(device)
            concede_lbl_dev = concede_lbl.to(device)
            score_logit, concede_logit = model(x_categ, x_cont)
            loss = score_criterion(score_logit, score_lbl_dev) + concede_criterion(concede_logit, concede_lbl_dev)
            total_loss += loss.item()
            score_prob = torch.sigmoid(score_logit).cpu().numpy()
            concede_prob = torch.sigmoid(concede_logit).cpu().numpy()
            for k in range(len(score_prob)):
                rows.append({
                    "match_id": int(match_ids[k]) if hasattr(match_ids[k], "item") else match_ids[k],
                    "source_name": source_names[k] if isinstance(source_names[k], str) else str(source_names[k]),
                    "action_index": int(action_indices[k]),
                    "score_label": int(score_lbl[k].item()),
                    "concede_label": int(concede_lbl[k].item()),
                    "score_prob": float(score_prob[k]),
                    "concede_prob": float(concede_prob[k]),
                })
    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
_EMPTY_COLS = ["match_id", "source_name", "action_index",
               "score_label", "concede_label", "score_prob", "concede_prob"]


def _empty_pred() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def _export(args, valid_splits, train_pred, val_pred, test_pred):
    if len(val_pred) > 0:
        score_thr, _ = find_best_threshold(
            val_pred["score_label"].astype(int).tolist(),
            val_pred["score_prob"].astype(float).tolist(),
        )
        concede_thr, _ = find_best_threshold(
            val_pred["concede_label"].astype(int).tolist(),
            val_pred["concede_prob"].astype(float).tolist(),
        )
    else:
        score_thr, concede_thr = 0.5, 0.5

    pred_by_split = {"train": train_pred, "val": val_pred, "test": test_pred}
    for s, df in pred_by_split.items():
        df["split"] = s

    all_pred = pd.concat([pred_by_split[s] for s in valid_splits], ignore_index=True)
    all_pred = add_vaep_columns(all_pred)

    if not args.include_source_name and "source_name" in all_pred.columns:
        all_pred = all_pred.drop(columns=["source_name"])

    output_path = choose_prediction_output_path(PREDICTIONS_OUTPUT, args.prediction_format)
    free_bytes = shutil.disk_usage(".").free
    if free_bytes < 2 * 1024**3:
        print(f"Warning: low disk space ({free_bytes/1024**3:.2f} GB free).")
    save_predictions_compact(all_pred, output_path, args.prediction_format)

    metrics_rows = []
    for split_name, df_split in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        if len(df_split) > 0:
            metrics_rows.extend(summarize_split(split_name, df_split, score_thr, concede_thr))
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved predictions: {output_path}")
    print(f"Saved metrics:     {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")


# ---------------------------------------------------------------------------
# Fast array-based dataset: pre-encode all dict lookups into numpy arrays once
# so __getitem__ is pure array indexing (10-20x faster than per-sample dict lookups)
# ---------------------------------------------------------------------------
class VAEPArrayDataset(Dataset):
    def __init__(self, cat_arr: np.ndarray, cont_arr: np.ndarray,
                 score_arr: np.ndarray, concede_arr: np.ndarray,
                 match_ids: list, source_names: list, action_indices: list):
        self.cat_arr = cat_arr
        self.cont_arr = cont_arr
        self.score_arr = score_arr
        self.concede_arr = concede_arr
        self.match_ids = match_ids
        self.source_names = source_names
        self.action_indices = action_indices

    def __len__(self) -> int:
        return len(self.score_arr)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.cat_arr[idx]).long(),
            torch.from_numpy(self.cont_arr[idx]),
            torch.tensor(self.score_arr[idx], dtype=torch.float32),
            torch.tensor(self.concede_arr[idx], dtype=torch.float32),
            self.match_ids[idx],
            self.source_names[idx],
            self.action_indices[idx],
        )


def actions_to_arrays(
    actions: list,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: dict,
    desc: str = "Pre-encoding",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Convert action dicts to numpy arrays once; much faster than per-sample dict lookups."""
    n = len(actions)
    cat_arr = np.zeros((n, len(cat_cols)), dtype=np.int32)
    cont_arr = np.zeros((n, len(cont_cols)), dtype=np.float32)
    score_arr = np.zeros(n, dtype=np.float32)
    concede_arr = np.zeros(n, dtype=np.float32)
    match_ids: list = [None] * n
    source_names: list = [None] * n
    action_indices: list = [None] * n

    for i, action in enumerate(tqdm(actions, desc=desc, leave=False)):
        feat = action["features"]
        for j, col in enumerate(cat_cols):
            cat_arr[i, j] = cat_maps[col].get(safe_text(feat.get(col)), 0)
        for j, col in enumerate(cont_cols):
            cont_arr[i, j] = safe_num(feat.get(col))
        score_arr[i] = float(action["score_label"])
        concede_arr[i] = float(action["concede_label"])
        match_ids[i] = int(action["match_id"])
        source_names[i] = str(action["source_name"])
        action_indices[i] = int(action["action_index"])

    return cat_arr, cont_arr, score_arr, concede_arr, match_ids, source_names, action_indices


def _make_loader_from_arrays(
    cat_arr, cont_arr, score_arr, concede_arr,
    match_ids, source_names, action_indices,
    batch_size: int, shuffle: bool, pin_memory: bool,
) -> DataLoader:
    ds = VAEPArrayDataset(cat_arr, cont_arr, score_arr, concede_arr,
                          match_ids, source_names, action_indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin_memory)


# ---------------------------------------------------------------------------
def main():
    global DEVICE
    args = parse_args()
    DEVICE = resolve_device(args.device)
    set_seed(SEED)
    args.checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    use_cache = not bool(args.max_train_matches)
    print(f"DEVICE = {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"PRETRAIN_CHECKPOINT = {args.pretrain_checkpoint}")
    print(f"CHECKPOINT_PATH = {args.checkpoint_path}")

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    # Load pretrained encoder (always needed)
    if not os.path.exists(args.pretrain_checkpoint):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_checkpoint}")
    print(f"Loading pretrained encoder from {args.pretrain_checkpoint}...")
    encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std = load_pretrained_encoder(
        args.pretrain_checkpoint
    )
    print(f"Encoder output dim = {encoder.output_dim}")

    use_pin = DEVICE.type == "cuda"

    # ------------------------------------------------------------------
    # Eval-only fast path
    # ------------------------------------------------------------------
    if args.eval_only:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for eval-only: {args.checkpoint_path}")
        print(f"Eval-only mode: loading checkpoint {args.checkpoint_path}")

        need_train = "train" in valid_splits
        need_val = "val" in valid_splits
        need_test = "test" in valid_splits

        if need_train or need_val:
            train_actions, val_actions = load_or_build_split_actions(
                TRAIN_ROOT, lookahead_k=10, val_ratio=VAL_RATIO, seed=SEED,
                max_match_files=args.max_train_matches or MAX_TRAIN_MATCH_FILES,
                use_cache=use_cache,
            )
            if not need_train:
                train_actions = []
        else:
            train_actions, val_actions = [], []

        test_actions = (
            load_or_build_actions(
                TEST_ROOT, lookahead_k=10,
                max_match_files=args.max_test_matches or MAX_TEST_MATCH_FILES,
                use_cache=use_cache,
            ) if need_test else []
        )
        print(f"train={len(train_actions)} | val={len(val_actions)} | test={len(test_actions)}")

        ckpt = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
        model = FrozenEncoderVAEP(encoder, HIDDEN_DIM).to(DEVICE)
        model.mlp.load_state_dict(ckpt["mlp_state_dict"])
        model.score_head.load_state_dict(ckpt["score_head_state_dict"])
        model.concede_head.load_state_dict(ckpt["concede_head_state_dict"])
        model.eval()

        crit = nn.BCEWithLogitsLoss()

        train_pred = _empty_pred()
        val_pred = _empty_pred()
        test_pred = _empty_pred()

        if need_train and train_actions:
            arrs = actions_to_arrays(train_actions, cat_cols, cont_cols, cat_maps, "Pre-encode train")
            _, train_pred = predict_dataset(
                model, _make_loader_from_arrays(*arrs, args.batch_size, False, use_pin),
                crit, crit, DEVICE, "Predict train")
        if need_val and val_actions:
            arrs = actions_to_arrays(val_actions, cat_cols, cont_cols, cat_maps, "Pre-encode val")
            _, val_pred = predict_dataset(
                model, _make_loader_from_arrays(*arrs, args.batch_size, False, use_pin),
                crit, crit, DEVICE, "Predict val")
        if need_test and test_actions:
            arrs = actions_to_arrays(test_actions, cat_cols, cont_cols, cat_maps, "Pre-encode test")
            _, test_pred = predict_dataset(
                model, _make_loader_from_arrays(*arrs, args.batch_size, False, use_pin),
                crit, crit, DEVICE, "Predict test")

        _export(args, valid_splits, train_pred, val_pred, test_pred)
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    train_actions, val_actions = load_or_build_split_actions(
        TRAIN_ROOT, lookahead_k=10, val_ratio=VAL_RATIO, seed=SEED,
        max_match_files=args.max_train_matches or MAX_TRAIN_MATCH_FILES,
        use_cache=use_cache,
    )
    test_actions = load_or_build_actions(
        TEST_ROOT, lookahead_k=10,
        max_match_files=args.max_test_matches or MAX_TEST_MATCH_FILES,
        use_cache=use_cache,
    )
    print(f"train={len(train_actions)} | val={len(val_actions)} | test={len(test_actions)}")

    # Pre-encode action dicts → numpy arrays (one-time cost, eliminates per-sample dict lookups)
    print("Pre-encoding splits to arrays...")
    train_arrs = actions_to_arrays(train_actions, cat_cols, cont_cols, cat_maps, "Pre-encode train")
    val_arrs   = actions_to_arrays(val_actions,   cat_cols, cont_cols, cat_maps, "Pre-encode val")
    test_arrs  = actions_to_arrays(test_actions,  cat_cols, cont_cols, cat_maps, "Pre-encode test")

    score_pos_weight   = compute_pos_weight(train_arrs[2].tolist())   # score_arr
    concede_pos_weight = compute_pos_weight(train_arrs[3].tolist())   # concede_arr

    train_loader = _make_loader_from_arrays(*train_arrs, args.batch_size, True,  use_pin)
    val_loader   = _make_loader_from_arrays(*val_arrs,   args.batch_size, False, use_pin)
    test_loader  = _make_loader_from_arrays(*test_arrs,  args.batch_size, False, use_pin)

    model = FrozenEncoderVAEP(encoder, HIDDEN_DIM).to(DEVICE)
    print(f"score_pos_weight={score_pos_weight:.4f} | concede_pos_weight={concede_pos_weight:.4f}")

    score_crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(score_pos_weight, dtype=torch.float32, device=DEVICE)
    )
    concede_crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(concede_pos_weight, dtype=torch.float32, device=DEVICE)
    )

    # Only optimize the MLP heads, not the frozen encoder
    trainable_params = list(model.mlp.parameters()) + \
                       list(model.score_head.parameters()) + \
                       list(model.concede_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(model, train_loader, optimizer, score_crit, concede_crit, DEVICE)
        val_loss, val_df = predict_dataset(model, val_loader, score_crit, concede_crit, DEVICE, "Val")

        score_ll = binary_log_loss(
            val_df["score_label"].astype(int).tolist(),
            val_df["score_prob"].astype(float).tolist(),
        )
        concede_ll = binary_log_loss(
            val_df["concede_label"].astype(int).tolist(),
            val_df["concede_prob"].astype(float).tolist(),
        )
        objective = 0.5 * (score_ll + concede_ll)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_score_ll={score_ll:.4f} | val_concede_ll={concede_ll:.4f}")
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}",
                              val_score_ll=f"{score_ll:.4f}",
                              val_concede_ll=f"{concede_ll:.4f}")

        if objective < best_val:
            best_val = objective
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "mlp_state_dict": model.mlp.state_dict(),
                    "score_head_state_dict": model.score_head.state_dict(),
                    "concede_head_state_dict": model.concede_head.state_dict(),
                    "embed_dim": encoder.output_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "pretrain_checkpoint": args.pretrain_checkpoint,
                    "epoch": int(epoch),
                },
                args.checkpoint_path,
            )
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    # Final evaluation
    ckpt = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
    model.mlp.load_state_dict(ckpt["mlp_state_dict"])
    model.score_head.load_state_dict(ckpt["score_head_state_dict"])
    model.concede_head.load_state_dict(ckpt["concede_head_state_dict"])

    eval_crit = nn.BCEWithLogitsLoss()

    train_pred = _empty_pred()
    if "train" in valid_splits:
        _, train_pred = predict_dataset(model, train_loader, eval_crit, eval_crit, DEVICE, "Predict train")

    _, val_pred = predict_dataset(model, val_loader, eval_crit, eval_crit, DEVICE, "Predict val")
    _, test_pred = predict_dataset(model, test_loader, eval_crit, eval_crit, DEVICE, "Predict test")

    _export(args, valid_splits, train_pred, val_pred, test_pred)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# resolve script dir so sibling imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    TabEventEncoder,
    build_event_features,
    set_seed,
)
from train_vaep_baseline import (
    CACHE_DIR,
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    SEED,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    add_vaep_columns,
    binary_log_loss,
    choose_prediction_output_path,
    compute_pos_weight,
    ensure_non_empty_split,
    find_best_threshold,
    get_cache_key,
    load_or_build_actions,
    load_or_build_split_actions,
    safe_num,
    safe_text,
    save_predictions_compact,
    split_train_val_by_match,
    summarize_split,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PRETRAIN_CHECKPOINT = os.path.join("checkpoints", "tabtransformer_row_mask_pretrain.pt")
CHECKPOINT_PATH = os.path.join("checkpoints", "vaep_tabtransformer_best.pt")
PREDICTIONS_OUTPUT = "vaep_tabtransformer_predictions"
METRICS_OUTPUT = "vaep_tabtransformer_metrics.csv"

EMBED_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")

HIDDEN_DIM = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAEP with TabTransformer embeddings")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    p.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--prediction-splits", default="val,test",
                   help="Comma-separated list of splits to export: train,val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches", type=int, default=None)
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_checkpoint_path(path: str) -> str:
    if not os.path.exists(path):
        basename = os.path.basename(path)
        candidate = os.path.join("checkpoints", basename)
        if os.path.exists(candidate):
            return candidate
    return path


# ---------------------------------------------------------------------------
# Embedding extraction & caching
# ---------------------------------------------------------------------------
def _checkpoint_hash(ckpt_path: str) -> str:
    """MD5 of the checkpoint file bytes for cache keying."""
    h = hashlib.md5()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _embed_cache_path(data_key: str, ckpt_hash: str) -> str:
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    combined = hashlib.md5(f"{data_key}_{ckpt_hash}".encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, f"embed_{combined}.pkl")


def build_encoder_from_checkpoint(ckpt_path: str) -> Tuple[TabEventEncoder, List[str], List[str], dict, List[int], np.ndarray]:
    """Load the pretrained TabEventEncoder and associated feature metadata."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cat_cols = list(ckpt["cat_cols"])
    cont_cols = list(ckpt["cont_cols"])
    cat_maps = ckpt["cat_maps"]
    cat_dims = list(ckpt["cat_dims"])
    cont_mean_std = np.asarray(ckpt["cont_mean_std"], dtype=np.float32)

    encoder = TabEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
    encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder.eval()
    return encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std


def extract_embeddings(
    actions: List[Dict],
    encoder: TabEventEncoder,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: dict,
    device: torch.device,
    batch_size: int = 2048,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """Run encoder in no_grad mode, return (N, embed_dim) float32 array."""
    encoder = encoder.to(device)
    encoder.eval()

    n = len(actions)
    embed_dim = encoder.output_dim
    result = np.zeros((n, embed_dim), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc=desc):
        batch = actions[start: start + batch_size]
        bsize = len(batch)

        x_categ = np.zeros((bsize, len(cat_cols)), dtype=np.int64)
        x_cont = np.zeros((bsize, len(cont_cols)), dtype=np.float32)

        for i, action in enumerate(batch):
            feat = action["features"]
            for j, col in enumerate(cat_cols):
                x_categ[i, j] = cat_maps[col].get(safe_text(feat.get(col)), 0)
            for j, col in enumerate(cont_cols):
                x_cont[i, j] = safe_num(feat.get(col))

        with torch.no_grad():
            emb = encoder(
                torch.tensor(x_categ, dtype=torch.long, device=device),
                torch.tensor(x_cont, dtype=torch.float32, device=device),
            )
        result[start: start + bsize] = emb.cpu().numpy()

    return result


def load_or_build_embeddings(
    actions: List[Dict],
    encoder: TabEventEncoder,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: dict,
    device: torch.device,
    data_cache_key: str,
    ckpt_hash: str,
    use_cache: bool = True,
    batch_size: int = 2048,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    cache_path = _embed_cache_path(data_cache_key, ckpt_hash)

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    embeddings = extract_embeddings(actions, encoder, cat_cols, cont_cols, cat_maps,
                                    device=device, batch_size=batch_size, desc=desc)

    if use_cache:
        print(f"Caching embeddings to {cache_path}...")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
        except (OSError, MemoryError) as exc:
            print(f"Warning: could not cache embeddings ({exc}). Continuing without cache.")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    return embeddings


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VAEPEmbeddingDataset(Dataset):
    def __init__(
        self,
        indices: List[int],
        all_embeddings: np.ndarray,
        all_actions: List[Dict],
    ):
        self.indices = indices
        self.embeddings = all_embeddings
        self.actions = all_actions

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        action = self.actions[i]
        emb = torch.tensor(self.embeddings[i], dtype=torch.float32)
        score_label = torch.tensor(float(action["score_label"]), dtype=torch.float32)
        concede_label = torch.tensor(float(action["concede_label"]), dtype=torch.float32)
        return emb, score_label, concede_label, action["match_id"], action["source_name"], int(i)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VAEPEmbeddingMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.concede_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.score_head(h).squeeze(-1), self.concede_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / predict helpers
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: VAEPEmbeddingMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for emb, score_lbl, concede_lbl, *_ in tqdm(loader, desc="Train", leave=False):
        emb = emb.to(device)
        score_lbl = score_lbl.to(device)
        concede_lbl = concede_lbl.to(device)
        optimizer.zero_grad()
        score_logit, concede_logit = model(emb)
        loss = score_criterion(score_logit, score_lbl) + concede_criterion(concede_logit, concede_lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def predict_dataset(
    model: VAEPEmbeddingMLP,
    loader: DataLoader,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
    desc: str = "Predict",
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows = []
    with torch.no_grad():
        for emb, score_lbl, concede_lbl, match_ids, source_names, action_indices in tqdm(loader, desc=desc, leave=False):
            emb = emb.to(device)
            score_lbl_dev = score_lbl.to(device)
            concede_lbl_dev = concede_lbl.to(device)
            score_logit, concede_logit = model(emb)
            loss = score_criterion(score_logit, score_lbl_dev) + concede_criterion(concede_logit, concede_lbl_dev)
            total_loss += loss.item()
            score_prob = torch.sigmoid(score_logit).cpu().numpy()
            concede_prob = torch.sigmoid(concede_logit).cpu().numpy()
            for k in range(len(score_prob)):
                rows.append({
                    "match_id": int(match_ids[k]) if hasattr(match_ids[k], "item") else match_ids[k],
                    "source_name": source_names[k] if isinstance(source_names[k], str) else str(source_names[k]),
                    "action_index": int(action_indices[k]),
                    "score_label": int(score_lbl[k].item()),
                    "concede_label": int(concede_lbl[k].item()),
                    "score_prob": float(score_prob[k]),
                    "concede_prob": float(concede_prob[k]),
                })
    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export helper (same pattern as vaep_baseline)
# ---------------------------------------------------------------------------
_EMPTY_COLS = ["match_id", "source_name", "action_index",
               "score_label", "concede_label", "score_prob", "concede_prob"]


def _empty_pred() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def _export(args, valid_splits, train_pred, val_pred, test_pred):
    if len(val_pred) > 0:
        score_thr, _ = find_best_threshold(
            val_pred["score_label"].astype(int).tolist(),
            val_pred["score_prob"].astype(float).tolist(),
        )
        concede_thr, _ = find_best_threshold(
            val_pred["concede_label"].astype(int).tolist(),
            val_pred["concede_prob"].astype(float).tolist(),
        )
    else:
        score_thr, concede_thr = 0.5, 0.5

    pred_by_split = {"train": train_pred, "val": val_pred, "test": test_pred}
    for s, df in pred_by_split.items():
        df["split"] = s

    all_pred = pd.concat([pred_by_split[s] for s in valid_splits], ignore_index=True)
    all_pred = add_vaep_columns(all_pred)

    if not args.include_source_name and "source_name" in all_pred.columns:
        all_pred = all_pred.drop(columns=["source_name"])

    output_path = choose_prediction_output_path(PREDICTIONS_OUTPUT, args.prediction_format)
    free_bytes = shutil.disk_usage(".").free
    if free_bytes < 2 * 1024**3:
        print(f"Warning: low disk space ({free_bytes/1024**3:.2f} GB free). "
              f"Exporting splits: {','.join(valid_splits)}")
    save_predictions_compact(all_pred, output_path, args.prediction_format)

    metrics_rows = []
    for split_name, df_split in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        if len(df_split) > 0:
            metrics_rows.extend(summarize_split(split_name, df_split, score_thr, concede_thr))
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved predictions: {output_path}")
    print(f"Saved metrics:     {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global DEVICE
    args = parse_args()
    DEVICE = resolve_device(args.device)
    set_seed(SEED)
    args.checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    use_cache = not bool(args.max_train_matches)
    print(f"DEVICE = {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"PRETRAIN_CHECKPOINT = {args.pretrain_checkpoint}")
    print(f"CHECKPOINT_PATH = {args.checkpoint_path}")

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    # ------------------------------------------------------------------
    # Load pretrained encoder (always needed to get feature metadata)
    # ------------------------------------------------------------------
    if not os.path.exists(args.pretrain_checkpoint):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_checkpoint}")

    print(f"Loading pretrained encoder from {args.pretrain_checkpoint}...")
    encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std = build_encoder_from_checkpoint(
        args.pretrain_checkpoint
    )
    ckpt_hash = _checkpoint_hash(args.pretrain_checkpoint)
    embed_dim = encoder.output_dim
    print(f"Encoder output dim = {embed_dim}")

    # ------------------------------------------------------------------
    # Eval-only fast path
    # ------------------------------------------------------------------
    if args.eval_only:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for eval-only: {args.checkpoint_path}")
        print(f"Eval-only mode: loading checkpoint {args.checkpoint_path}")

        need_train = "train" in valid_splits
        need_val = "val" in valid_splits
        need_test = "test" in valid_splits
        use_pin = DEVICE.type == "cuda"

        if need_train or need_val:
            train_actions, val_actions = load_or_build_split_actions(
                TRAIN_ROOT, lookahead_k=10, val_ratio=VAL_RATIO, seed=SEED,
                max_match_files=args.max_train_matches or MAX_TRAIN_MATCH_FILES,
                use_cache=use_cache,
            )
            if not need_train:
                train_actions = []
        else:
            train_actions, val_actions = [], []

        test_actions = (
            load_or_build_actions(
                TEST_ROOT, lookahead_k=10,
                max_match_files=args.max_test_matches or MAX_TEST_MATCH_FILES,
                use_cache=use_cache,
            ) if need_test else []
        )

        print(f"train={len(train_actions)} | val={len(val_actions)} | test={len(test_actions)}")

        # Embedding extraction (each split independently, with its own cache key)
        def _get_embeds(actions, tag):
            if not actions:
                return np.zeros((0, embed_dim), dtype=np.float32)
            base_key = get_cache_key(
                TRAIN_ROOT if tag != "test" else TEST_ROOT,
                10,
                args.max_train_matches or MAX_TRAIN_MATCH_FILES
                if tag != "test"
                else args.max_test_matches or MAX_TEST_MATCH_FILES,
            )
            return load_or_build_embeddings(
                actions, encoder, cat_cols, cont_cols, cat_maps, DEVICE,
                data_cache_key=f"{base_key}_{tag}",
                ckpt_hash=ckpt_hash,
                use_cache=use_cache,
                desc=f"Embed {tag}",
            )

        train_emb = _get_embeds(train_actions, "train")
        val_emb = _get_embeds(val_actions, "val")
        test_emb = _get_embeds(test_actions, "test")

        all_actions = list(train_actions) + list(val_actions) + list(test_actions)
        all_emb = np.concatenate([train_emb, val_emb, test_emb], axis=0) if len(all_actions) > 0 else np.zeros((0, embed_dim), dtype=np.float32)

        t_idx = list(range(len(train_actions)))
        v_idx = list(range(len(train_actions), len(train_actions) + len(val_actions)))
        e_idx = list(range(len(train_actions) + len(val_actions), len(all_actions)))

        ckpt = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
        model = VAEPEmbeddingMLP(embed_dim, args.__dict__.get("hidden_dim", HIDDEN_DIM)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        score_crit = nn.BCEWithLogitsLoss()
        concede_crit = nn.BCEWithLogitsLoss()

        def _make_loader(idx, actions, emb):
            ds = VAEPEmbeddingDataset(idx, emb, actions)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_pin)

        train_pred = _empty_pred()
        val_pred = _empty_pred()
        test_pred = _empty_pred()

        if need_train and train_actions:
            _, train_pred = predict_dataset(model, _make_loader(t_idx, all_actions, all_emb),
                                            score_crit, concede_crit, DEVICE, "Predict train")
        if need_val and val_actions:
            _, val_pred = predict_dataset(model, _make_loader(v_idx, all_actions, all_emb),
                                          score_crit, concede_crit, DEVICE, "Predict val")
        if need_test and test_actions:
            _, test_pred = predict_dataset(model, _make_loader(e_idx, all_actions, all_emb),
                                           score_crit, concede_crit, DEVICE, "Predict test")

        _export(args, valid_splits, train_pred, val_pred, test_pred)
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    train_actions, val_actions = load_or_build_split_actions(
        TRAIN_ROOT, lookahead_k=10, val_ratio=VAL_RATIO, seed=SEED,
        max_match_files=args.max_train_matches or MAX_TRAIN_MATCH_FILES,
        use_cache=use_cache,
    )
    test_actions = load_or_build_actions(
        TEST_ROOT, lookahead_k=10,
        max_match_files=args.max_test_matches or MAX_TEST_MATCH_FILES,
        use_cache=use_cache,
    )
    print(f"train={len(train_actions)} | val={len(val_actions)} | test={len(test_actions)}")

    # Extract embeddings for each split (cached separately)
    train_key = get_cache_key(TRAIN_ROOT, 10, args.max_train_matches or MAX_TRAIN_MATCH_FILES)
    test_key = get_cache_key(TEST_ROOT, 10, args.max_test_matches or MAX_TEST_MATCH_FILES)

    train_emb = load_or_build_embeddings(
        train_actions, encoder, cat_cols, cont_cols, cat_maps, DEVICE,
        data_cache_key=f"{train_key}_train", ckpt_hash=ckpt_hash,
        use_cache=use_cache, desc="Embed train",
    )
    val_emb = load_or_build_embeddings(
        val_actions, encoder, cat_cols, cont_cols, cat_maps, DEVICE,
        data_cache_key=f"{train_key}_val", ckpt_hash=ckpt_hash,
        use_cache=use_cache, desc="Embed val",
    )
    test_emb = load_or_build_embeddings(
        test_actions, encoder, cat_cols, cont_cols, cat_maps, DEVICE,
        data_cache_key=f"{test_key}_test", ckpt_hash=ckpt_hash,
        use_cache=use_cache, desc="Embed test",
    )

    # Merge for Dataset indexing
    all_actions = list(train_actions) + list(val_actions) + list(test_actions)
    all_emb = np.concatenate([train_emb, val_emb, test_emb], axis=0)

    t_idx = list(range(len(train_actions)))
    v_idx = list(range(len(train_actions), len(train_actions) + len(val_actions)))
    e_idx = list(range(len(train_actions) + len(val_actions), len(all_actions)))

    use_pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        VAEPEmbeddingDataset(t_idx, all_emb, all_actions),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        VAEPEmbeddingDataset(v_idx, all_emb, all_actions),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        VAEPEmbeddingDataset(e_idx, all_emb, all_actions),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin,
    )

    model = VAEPEmbeddingMLP(embed_dim, HIDDEN_DIM).to(DEVICE)

    score_pos_weight = compute_pos_weight([int(a["score_label"]) for a in train_actions])
    concede_pos_weight = compute_pos_weight([int(a["concede_label"]) for a in train_actions])
    print(f"score_pos_weight={score_pos_weight:.4f} | concede_pos_weight={concede_pos_weight:.4f}")

    score_crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(score_pos_weight, dtype=torch.float32, device=DEVICE)
    )
    concede_crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(concede_pos_weight, dtype=torch.float32, device=DEVICE)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(model, train_loader, optimizer, score_crit, concede_crit, DEVICE)
        val_loss, val_df = predict_dataset(model, val_loader, score_crit, concede_crit, DEVICE, "Val")

        score_ll = binary_log_loss(
            val_df["score_label"].astype(int).tolist(),
            val_df["score_prob"].astype(float).tolist(),
        )
        concede_ll = binary_log_loss(
            val_df["concede_label"].astype(int).tolist(),
            val_df["concede_prob"].astype(float).tolist(),
        )
        objective = 0.5 * (score_ll + concede_ll)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_score_ll={score_ll:.4f} | val_concede_ll={concede_ll:.4f}")
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}",
                              val_score_ll=f"{score_ll:.4f}",
                              val_concede_ll=f"{concede_ll:.4f}")

        if objective < best_val:
            best_val = objective
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embed_dim": embed_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "pretrain_checkpoint": args.pretrain_checkpoint,
                    "epoch": int(epoch),
                },
                args.checkpoint_path,
            )
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    # Final evaluation on best checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_crit = nn.BCEWithLogitsLoss()

    train_pred = _empty_pred()
    if "train" in valid_splits:
        _, train_pred = predict_dataset(model, train_loader, eval_crit, eval_crit, DEVICE, "Predict train")

    _, val_pred = predict_dataset(model, val_loader, eval_crit, eval_crit, DEVICE, "Predict val")
    _, test_pred = predict_dataset(model, test_loader, eval_crit, eval_crit, DEVICE, "Predict test")

    _export(args, valid_splits, train_pred, val_pred, test_pred)


if __name__ == "__main__":
    main()
