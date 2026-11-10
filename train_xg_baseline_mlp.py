import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_xg_tabtransformer import (
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    WEIGHT_MODE,
    MAX_POS_WEIGHT,
    binary_f1_score,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    find_best_threshold,
    load_sequences_from_root,
    safe_num,
    safe_text,
    set_seed,
    split_train_val_by_match,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


def with_exp_suffix(filename: str) -> str:
    exp_id = os.getenv("EXPERIMENT_ID", "").strip()
    if not exp_id:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{exp_id}{ext}"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BATCH_SIZE = 256
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN_DIM = 256
USE_CLASS_WEIGHTS = True

BASELINE_CHECKPOINT = os.path.join("checkpoints", with_exp_suffix("xg_baseline_mlp_best.pt"))
BASELINE_METRICS_OUTPUT = with_exp_suffix("xg_baseline_mlp_metrics.csv")
BASELINE_PREDICTIONS_OUTPUT = with_exp_suffix("xg_baseline_mlp_predictions.csv")


def build_terminal_event(seq: Dict[str, object]) -> Dict[str, object]:
    events = seq["events"]
    if not events:
        return {}
    return events[-1]


def build_cat_maps_from_terminal_events(
    sequences: List[Dict[str, object]], cat_cols: List[str]
) -> Tuple[Dict[str, Dict[str, int]], List[int]]:
    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        uniques = sorted({safe_text(build_terminal_event(seq).get(col)) for seq in sequences})
        cat_maps[col] = {value: idx + 1 for idx, value in enumerate(uniques)}
    cat_dims = [len(cat_maps[col]) + 1 for col in cat_cols]
    return cat_maps, cat_dims


def compute_cont_stats_from_terminal_events(
    sequences: List[Dict[str, object]], cont_cols: List[str]
) -> np.ndarray:
    stats = []
    for col in cont_cols:
        values = [safe_num(build_terminal_event(seq).get(col)) for seq in sequences]
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean()) if len(arr) > 0 else 0.0
        std = float(arr.std()) if len(arr) > 1 else 1.0
        if std < 1e-6:
            std = 1.0
        stats.append((mean, std))
    return np.asarray(stats, dtype=np.float32)


class TerminalShotDataset(Dataset):
    def __init__(
        self,
        sequences: List[Dict[str, object]],
        cat_cols: List[str],
        cont_cols: List[str],
        cat_maps: Dict[str, Dict[str, int]],
    ):
        self.sequences = sequences
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.cat_maps = cat_maps

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sample = self.sequences[idx]
        event = build_terminal_event(sample)

        cat_values = [self.cat_maps[col].get(safe_text(event.get(col)), 0) for col in self.cat_cols]
        cont_values = [safe_num(event.get(col)) for col in self.cont_cols]

        return (
            torch.tensor(cat_values, dtype=torch.long),
            torch.tensor(cont_values, dtype=torch.float32),
            torch.tensor(int(sample["label"]), dtype=torch.long),
            int(sample["match_id"]),
            str(sample["source_name"]),
            int(sample["sequence_id"]),
        )


class TabMLPBaseline(nn.Module):
    def __init__(self, cat_dims: List[int], num_cont: int, cont_mean_std: np.ndarray):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_cont = num_cont

        emb_dims = [max(4, min(32, int(round(1.6 * (dim ** 0.56))))) for dim in cat_dims]
        self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim, emb_dim in zip(cat_dims, emb_dims)])

        mean_std = torch.tensor(cont_mean_std, dtype=torch.float32)
        self.register_buffer("cont_mean", mean_std[:, 0])
        self.register_buffer("cont_std", mean_std[:, 1])

        in_dim = int(sum(emb_dims) + num_cont)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 2),
        )

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x_categ[:, i]))
        x_cat = torch.cat(embs, dim=-1) if embs else torch.empty((x_categ.shape[0], 0), device=x_categ.device)

        x_cont_norm = (x_cont - self.cont_mean) / self.cont_std
        x = torch.cat([x_cat, x_cont_norm], dim=-1)
        return self.mlp(x)


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    if n_pos == 0 or not USE_CLASS_WEIGHTS:
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE)

    ratio = n_neg / max(n_pos, 1)
    if WEIGHT_MODE == "ratio":
        pos_weight = ratio
    elif WEIGHT_MODE == "capped":
        pos_weight = min(MAX_POS_WEIGHT, ratio)
    else:
        pos_weight = min(MAX_POS_WEIGHT, math.sqrt(ratio))

    return torch.tensor([1.0, float(pos_weight)], dtype=torch.float32, device=DEVICE)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows = []

    with torch.no_grad():
        for x_categ, x_cont, labels, match_ids, source_names, sequence_ids in loader:
            x_categ = x_categ.to(DEVICE)
            x_cont = x_cont.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(x_categ, x_cont)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            total_loss += float(loss.item())
            for i in range(len(labels)):
                rows.append(
                    {
                        "match_id": int(match_ids[i]),
                        "source_name": str(source_names[i]),
                        "sequence_id": int(sequence_ids[i]),
                        "label": int(labels[i].cpu().item()),
                        "goal_prob": float(probs[i].cpu().item()),
                    }
                )

    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


def evaluate_prediction_df(pred_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    preds = (pred_df["goal_prob"].values >= threshold).astype(int)
    labels = pred_df["label"].astype(int).tolist()
    probs = pred_df["goal_prob"].astype(float).tolist()
    return {
        "threshold": threshold,
        "f1": binary_f1_score(labels, preds.tolist()),
        "roc_auc": binary_roc_auc(labels, probs),
        "log_loss": binary_log_loss(labels, probs),
        "brier": brier_score(labels, probs),
        "positive_rate": float(pred_df["label"].mean()) if len(pred_df) > 0 else 0.0,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0

    for x_categ, x_cont, labels, _, _, _ in tqdm(loader, desc="Train", leave=False):
        x_categ = x_categ.to(DEVICE)
        x_cont = x_cont.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(x_categ, x_cont)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def main() -> None:
    set_seed(SEED)
    print(f"DEVICE = {DEVICE}")
    print(f"BASELINE_CHECKPOINT={BASELINE_CHECKPOINT}")

    # Use the same feature schema as xG task to keep comparison fair.
    pretrain_meta_path = os.path.join("checkpoints", with_exp_suffix("tabtransformer_row_mask_pretrain.pt"))
    if os.path.exists(pretrain_meta_path):
        meta_ckpt = torch.load(pretrain_meta_path, map_location="cpu", weights_only=False)
        print(f"Using pretrain metadata: {pretrain_meta_path}")
        cat_cols = list(meta_ckpt["cat_cols"])
        cont_cols = list(meta_ckpt["cont_cols"])
    else:
        # G0 and similar no-pretrain settings do not produce a pretrain metadata file.
        # Fall back to the default xG feature schema so baseline can still run.
        print("未找到预训练元数据，将使用默认 xG 特征定义。")
        cat_cols = list(DEFAULT_CAT_COLS)
        cont_cols = list(DEFAULT_CONT_COLS)

    train_sequences_all = load_sequences_from_root(
        TRAIN_ROOT, cat_cols, cont_cols, max_match_files=MAX_TRAIN_MATCH_FILES
    )
    test_sequences = load_sequences_from_root(
        TEST_ROOT, cat_cols, cont_cols, max_match_files=MAX_TEST_MATCH_FILES
    )
    train_sequences, val_sequences = split_train_val_by_match(train_sequences_all, val_ratio=VAL_RATIO, seed=SEED)

    cat_maps, cat_dims = build_cat_maps_from_terminal_events(train_sequences, cat_cols)
    cont_mean_std = compute_cont_stats_from_terminal_events(train_sequences, cont_cols)

    print(f"训练 shots: {len(train_sequences)} | 验证 shots: {len(val_sequences)} | 测试 shots: {len(test_sequences)}")
    train_labels = [int(seq["label"]) for seq in train_sequences]
    print(f"训练集进球率 = {np.mean(train_labels):.4f}")

    train_loader = DataLoader(
        TerminalShotDataset(train_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TerminalShotDataset(val_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TerminalShotDataset(test_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TabMLPBaseline(cat_dims, len(cont_cols), cont_mean_std).to(DEVICE)

    class_weights = compute_class_weights(train_labels)
    print(f"Class weights = {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(os.path.dirname(BASELINE_CHECKPOINT), exist_ok=True)
    best_val_auc = -1.0
    best_threshold = 0.5

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_pred_df = evaluate_model(model, val_loader, criterion)
        current_threshold, _ = find_best_threshold(val_pred_df["label"].tolist(), val_pred_df["goal_prob"].tolist())
        val_metrics = evaluate_prediction_df(val_pred_df, threshold=current_threshold)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_roc_auc={val_metrics['roc_auc']:.4f} | "
            f"val_logloss={val_metrics['log_loss']:.4f} | best_thr={current_threshold:.2f}"
        )

        val_auc = val_metrics["roc_auc"]
        if np.isnan(val_auc):
            val_auc = -1.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = current_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "cat_maps": cat_maps,
                    "cat_dims": cat_dims,
                    "cont_mean_std": cont_mean_std,
                    "best_threshold": best_threshold,
                },
                BASELINE_CHECKPOINT,
            )
            print(f"已保存最佳 baseline 模型: {BASELINE_CHECKPOINT}")

    if os.path.exists(BASELINE_CHECKPOINT):
        best_ckpt = torch.load(BASELINE_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_threshold = float(best_ckpt.get("best_threshold", 0.5))

    test_loss, test_pred_df = evaluate_model(model, test_loader, criterion)
    test_metrics = evaluate_prediction_df(test_pred_df, threshold=best_threshold)
    test_pred_df["pred"] = (test_pred_df["goal_prob"].values >= best_threshold).astype(int)

    print(
        f"\nOverall Test | loss={test_loss:.4f} | f1={test_metrics['f1']:.4f} | "
        f"roc_auc={test_metrics['roc_auc']:.4f} | logloss={test_metrics['log_loss']:.4f} | "
        f"brier={test_metrics['brier']:.4f} | threshold={best_threshold:.2f}"
    )

    rows = [
        {
            "split": "overall_test",
            "source_name": "ALL",
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "log_loss": test_metrics["log_loss"],
            "brier": test_metrics["brier"],
            "n_shots": len(test_pred_df),
            "positive_rate": test_metrics["positive_rate"],
            "threshold": best_threshold,
        }
    ]

    for source_name, group in test_pred_df.groupby("source_name"):
        group_metrics = evaluate_prediction_df(group, threshold=best_threshold)
        rows.append(
            {
                "split": "heldout_tournament",
                "source_name": source_name,
                "f1": group_metrics["f1"],
                "roc_auc": group_metrics["roc_auc"],
                "log_loss": group_metrics["log_loss"],
                "brier": group_metrics["brier"],
                "n_shots": len(group),
                "positive_rate": group_metrics["positive_rate"],
                "threshold": best_threshold,
            }
        )
        print(
            f"{source_name} | n={len(group)} | goal_rate={group_metrics['positive_rate']:.4f} | "
            f"F1={group_metrics['f1']:.4f} | ROC-AUC={group_metrics['roc_auc']:.4f} | "
            f"LogLoss={group_metrics['log_loss']:.4f}"
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(BASELINE_METRICS_OUTPUT, index=False)
    test_pred_df.to_csv(BASELINE_PREDICTIONS_OUTPUT, index=False)
    print(f"\nBaseline 评估结果已保存到: {BASELINE_METRICS_OUTPUT}")
    print(f"Baseline 每个 shot 的预测已保存到: {BASELINE_PREDICTIONS_OUTPUT}")


if __name__ == "__main__":
    main()
