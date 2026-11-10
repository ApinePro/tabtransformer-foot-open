"""
run_xg_kfold.py — xG 任务的 match-level K-fold 交叉验证 runner。

模型（--models 逗号分隔）:
  mlp        : MLP baseline（仅终末射门事件，随机 emb 端到端）
  gru_basic  : 随机 emb 端到端 + 2 层 GRU
  gru_frozen : 冻结预训练 TabEventEncoder + 2 层 GRU
  tt_frozen  : 冻结预训练 TabEventEncoder + 2 层 Transformer（G2F 同构）

协议（全模型统一）:
  - K=5 match-level fold，FOLD_SEED=42（fold 划分与训练 seed 解耦）
  - 每折训练 seed = 1000 + fold_index（同折内所有模型共用 → 配对设计）
  - EPOCHS=20，best-val-AUC 选 checkpoint，val 上选 F1 阈值
  - test 固定为 dataset-test 三赛事，不参与 fold
  - 其余超参沿用各模型脚本的既有默认值

抗中断:
  - fold 划分表、序列 pickle 缓存先落盘；每个 (model, fold) 单元完成即写结果
  - 重启自动跳过已完成单元；所有写入均为 tmp+os.replace 原子操作
  - 单元级 try/except 隔离，错误写 *_ERROR.txt 后继续

用法:
  python run_xg_kfold.py --models mlp,gru_basic,gru_frozen     # Phase 1
  python run_xg_kfold.py --models tt_frozen                    # Phase 2（复用同一 fold 划分）
  python run_xg_kfold.py --aggregate-only                      # 只重算汇总表
"""

import argparse
import copy
import gc
import glob
import hashlib
import math
import os
import pickle
import random
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    BATCH_SIZE as SEQ_BATCH_SIZE,
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    DEVICE,
    DROPOUT as XG_DROPOUT,
    HIDDEN_DIM as XG_HIDDEN_DIM,
    LR,
    MAX_SEQ_LEN,
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    NUM_SEQUENCE_HEADS,
    NUM_SEQUENCE_LAYERS,
    PRETRAIN_CHECKPOINT,
    TEST_ROOT,
    TRAIN_ROOT,
    WEIGHT_DECAY,
    SequenceXGModel,
    ShotSequenceDataset,
    TabEventEncoder,
    build_cat_maps_from_sequences,
    compute_class_weights,
    compute_cont_stats,
    evaluate_prediction_df,
    find_best_threshold,
    load_sequences_from_root,
    maybe_load_pretrain_metadata,
    predict_model,
    set_seed,
    train_one_epoch,
)
from train_xg_baseline_mlp import (
    BATCH_SIZE as MLP_BATCH_SIZE,
    LR as MLP_LR,
    TabMLPBaseline,
    TerminalShotDataset,
    build_cat_maps_from_terminal_events,
    compute_cont_stats_from_terminal_events,
)
from train_xg_gru import BasicEventEncoder, GRUXGModel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

# ---------------------------------------------------------------------------
K_FOLDS         = 5
FOLD_SEED       = 42
TRAIN_SEED_BASE = 1000
KFOLD_EPOCHS    = 20

FOLD_ASSIGN_CSV = "xg_kfold_fold_assignments.csv"
RESULTS_DIR     = "kfold_results"
COMBINED_METRICS = "xg_kfold_metrics.csv"
SUMMARY_CSV      = "xg_kfold_summary.csv"
SEQ_CACHE_DIR    = os.path.join(".vaep_cache", "xg_kfold")

ALL_MODELS = ["mlp", "mlp_frozen", "gru_basic", "gru_frozen", "tt_basic", "tt_frozen"]
METRIC_COLS = ["f1", "roc_auc", "log_loss", "brier"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match-level K-fold CV for xG models")
    p.add_argument("--models", default="mlp,gru_basic,gru_frozen",
                   help=f"Comma-separated subset of {ALL_MODELS}")
    p.add_argument("--select-metric", default="auc", choices=["auc", "brier"],
                   help="Validation metric for best-checkpoint selection; brier variants get '_bsel' suffix")
    p.add_argument("--epochs", type=int, default=KFOLD_EPOCHS)
    p.add_argument("--k", type=int, default=K_FOLDS)
    p.add_argument("--folds", default=None, help="e.g. '0,2,4' to run specific folds only")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--no-seq-cache", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None, help="Smoke test only")
    p.add_argument("--max-test-matches", type=int, default=None, help="Smoke test only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Atomic IO helpers
# ---------------------------------------------------------------------------
def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    compression = "gzip" if path.endswith(".gz") else None
    df.to_csv(tmp, index=False, encoding="utf-8-sig", compression=compression)
    os.replace(tmp, path)


def cell_paths(model: str, fold: int) -> Tuple[str, str, str]:
    base = os.path.join(RESULTS_DIR, f"xg_kfold_{model}_fold{fold}")
    return base + "_metrics.csv", base + "_predictions.csv.gz", base + "_ERROR.txt"


# ---------------------------------------------------------------------------
# Sequence loading with pickle cache (restart resilience)
# ---------------------------------------------------------------------------
def load_all_sequences(cat_cols, cont_cols, max_train, max_test, use_cache: bool):
    os.makedirs(SEQ_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(
        f"{TRAIN_ROOT}|{TEST_ROOT}|{','.join(cat_cols)}|{','.join(cont_cols)}|{max_train}|{max_test}".encode()
    ).hexdigest()[:12]
    cache_path = os.path.join(SEQ_CACHE_DIR, f"sequences_{key}.pkl")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached sequences from {cache_path} ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    train_all = load_sequences_from_root(TRAIN_ROOT, cat_cols, cont_cols, max_match_files=max_train)
    test_seqs = load_sequences_from_root(TEST_ROOT, cat_cols, cont_cols, max_match_files=max_test)

    if use_cache:
        tmp = cache_path + ".tmp"
        print(f"Caching sequences to {cache_path} ...")
        try:
            with open(tmp, "wb") as f:
                pickle.dump((train_all, test_seqs), f, protocol=4)
            os.replace(tmp, cache_path)
        except (OSError, MemoryError) as exc:
            print(f"Warning: sequence cache failed ({exc}); continuing without cache.")
            try:
                os.remove(tmp)
            except OSError:
                pass
    return train_all, test_seqs


# ---------------------------------------------------------------------------
# Fold assignment (persisted, shared by all models/phases)
# ---------------------------------------------------------------------------
def get_fold_assignments(train_sequences: list, k: int) -> Dict[int, int]:
    if os.path.exists(FOLD_ASSIGN_CSV):
        df = pd.read_csv(FOLD_ASSIGN_CSV)
        print(f"Loaded existing fold assignments: {FOLD_ASSIGN_CSV} "
              f"({df['match_id'].nunique()} matches, k={df['fold'].nunique()})")
        return dict(zip(df["match_id"].astype(int), df["fold"].astype(int)))

    match_ids = sorted({int(seq["match_id"]) for seq in train_sequences})
    rng = random.Random(FOLD_SEED)
    rng.shuffle(match_ids)
    chunks = np.array_split(np.asarray(match_ids), k)
    assign = {int(mid): fold for fold, chunk in enumerate(chunks) for mid in chunk}

    df = pd.DataFrame(sorted(assign.items()), columns=["match_id", "fold"])
    atomic_write_csv(df, FOLD_ASSIGN_CSV)
    print(f"Created fold assignments (FOLD_SEED={FOLD_SEED}, k={k}) → {FOLD_ASSIGN_CSV}")
    return assign


def split_fold(train_sequences: list, assign: Dict[int, int], fold: int):
    train = [s for s in train_sequences if assign.get(int(s["match_id"]), -1) != fold]
    val   = [s for s in train_sequences if assign.get(int(s["match_id"]), -1) == fold]
    return train, val


# ---------------------------------------------------------------------------
# MLP-specific train/eval (signature: model(x_cat, x_cont), no mask)
# ---------------------------------------------------------------------------
def mlp_train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    for x_cat, x_cont, labels, _, _, _ in loader:
        x_cat, x_cont, labels = x_cat.to(DEVICE), x_cont.to(DEVICE), labels.to(DEVICE)
        logits = model(x_cat, x_cont)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def mlp_predict(model, loader, criterion) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total = 0.0
    rows = []
    with torch.no_grad():
        for x_cat, x_cont, labels, match_ids, source_names, sequence_ids in loader:
            x_cat, x_cont, labels = x_cat.to(DEVICE), x_cont.to(DEVICE), labels.to(DEVICE)
            logits = model(x_cat, x_cont)
            total += float(criterion(logits, labels).item())
            probs = torch.softmax(logits, dim=-1)[:, 1]
            for i in range(len(labels)):
                rows.append({
                    "match_id": int(match_ids[i]),
                    "source_name": str(source_names[i]),
                    "sequence_id": int(sequence_ids[i]),
                    "label": int(labels[i].cpu().item()),
                    "goal_prob": float(probs[i].cpu().item()),
                })
    return total / max(len(loader), 1), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model factories — each returns (model, train_fn, predict_fn, loaders)
# ---------------------------------------------------------------------------
class BasicTransformerXG(nn.Module):
    """basic 随机 emb 事件编码 + SequenceXGModel 同构 Transformer 序列 encoder（CLS+位置编码+mask）。"""

    def __init__(self, event_encoder):
        super().__init__()
        self.event_encoder = event_encoder
        self.event_proj = nn.Sequential(
            nn.Linear(event_encoder.output_dim, XG_HIDDEN_DIM), nn.ReLU(), nn.Dropout(XG_DROPOUT))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, XG_HIDDEN_DIM))
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN + 1, XG_HIDDEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=XG_HIDDEN_DIM, nhead=NUM_SEQUENCE_HEADS, dim_feedforward=XG_HIDDEN_DIM * 4,
            dropout=XG_DROPOUT, batch_first=True, activation="gelu")
        self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=NUM_SEQUENCE_LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(XG_HIDDEN_DIM), nn.Linear(XG_HIDDEN_DIM, 2))

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_cat = x_categ.shape
        feats = self.event_encoder(x_categ.view(batch_size * seq_len, num_cat),
                                   x_cont.view(batch_size * seq_len, -1))
        tokens = self.event_proj(feats).view(batch_size, seq_len, -1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : seq_len + 1]
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        key_padding_mask = ~torch.cat([cls_mask, valid_mask], dim=1)
        x = self.sequence_encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(x[:, 0])


def load_frozen_encoder(cat_dims, num_cont, cont_mean_std) -> TabEventEncoder:
    encoder = TabEventEncoder(cat_dims, num_cont, cont_mean_std)
    ckpt = torch.load(PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    encoder.backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


class FrozenTerminalMLP(nn.Module):
    """冻结encoder编码终末射门事件(911维) + TabMLPBaseline同构MLP头。"""

    def __init__(self, encoder: TabEventEncoder):
        super().__init__()
        self.event_encoder = encoder
        from train_xg_baseline_mlp import DROPOUT as MLP_DROPOUT, HIDDEN_DIM as MLP_HIDDEN
        self.mlp = nn.Sequential(
            nn.Linear(encoder.output_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN // 2, 2),
        )

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.event_encoder(x_categ, x_cont))


def build_cell(model_name: str, fold_train, fold_val, test_seqs, pretrain_meta):
    """Build model + loaders for one (model, fold) cell. Fold-train-only statistics."""
    if pretrain_meta is not None:
        cat_cols = list(pretrain_meta["cat_cols"])
        cont_cols = list(pretrain_meta["cont_cols"])
    else:
        cat_cols = list(DEFAULT_CAT_COLS)
        cont_cols = list(DEFAULT_CONT_COLS)

    if model_name == "mlp":
        cat_maps, cat_dims = build_cat_maps_from_terminal_events(fold_train, cat_cols)
        cont_mean_std = compute_cont_stats_from_terminal_events(fold_train, cont_cols)
        model = TabMLPBaseline(cat_dims, len(cont_cols), cont_mean_std).to(DEVICE)

        def make_loader(seqs, shuffle):
            return DataLoader(TerminalShotDataset(seqs, cat_cols, cont_cols, cat_maps),
                              batch_size=MLP_BATCH_SIZE, shuffle=shuffle, num_workers=0)
        return model, mlp_train_one_epoch, mlp_predict, make_loader

    if model_name == "mlp_frozen":
        if pretrain_meta is None:
            raise FileNotFoundError(f"Pretrain checkpoint required: {PRETRAIN_CHECKPOINT}")
        cat_maps = pretrain_meta["cat_maps"]
        cat_dims = list(pretrain_meta["cat_dims"])
        cont_mean_std = np.asarray(pretrain_meta["cont_mean_std"], dtype=np.float32)
        encoder = load_frozen_encoder(cat_dims, len(cont_cols), cont_mean_std)
        model = FrozenTerminalMLP(encoder).to(DEVICE)

        def make_loader(seqs, shuffle):
            return DataLoader(TerminalShotDataset(seqs, cat_cols, cont_cols, cat_maps),
                              batch_size=MLP_BATCH_SIZE, shuffle=shuffle, num_workers=0)
        return model, mlp_train_one_epoch, mlp_predict, make_loader

    if model_name == "gru_basic":
        cat_maps, cat_dims = build_cat_maps_from_sequences(fold_train, cat_cols)
        cont_mean_std = compute_cont_stats(fold_train, cont_cols)
        encoder = BasicEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
        model = GRUXGModel(encoder, encoder.output_dim).to(DEVICE)

    elif model_name == "tt_basic":
        cat_maps, cat_dims = build_cat_maps_from_sequences(fold_train, cat_cols)
        cont_mean_std = compute_cont_stats(fold_train, cont_cols)
        encoder = BasicEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
        model = BasicTransformerXG(encoder).to(DEVICE)

    elif model_name in ("gru_frozen", "tt_frozen"):
        if pretrain_meta is None:
            raise FileNotFoundError(f"Pretrain checkpoint required: {PRETRAIN_CHECKPOINT}")
        cat_maps = pretrain_meta["cat_maps"]
        cat_dims = list(pretrain_meta["cat_dims"])
        cont_mean_std = np.asarray(pretrain_meta["cont_mean_std"], dtype=np.float32)

        if model_name == "gru_frozen":
            encoder = load_frozen_encoder(cat_dims, len(cont_cols), cont_mean_std)
            model = GRUXGModel(encoder, encoder.output_dim).to(DEVICE)
        else:
            model = SequenceXGModel(cat_dims, len(cont_cols), cont_mean_std)
            ckpt = torch.load(PRETRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
            model.event_encoder.backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
            for p in model.event_encoder.parameters():
                p.requires_grad = False
            model = model.to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    def make_loader(seqs, shuffle):
        return DataLoader(ShotSequenceDataset(seqs, cat_cols, cont_cols, cat_maps),
                          batch_size=SEQ_BATCH_SIZE, shuffle=shuffle, num_workers=0)
    return model, train_one_epoch, predict_model, make_loader


# ---------------------------------------------------------------------------
def run_cell(model_name: str, fold: int, fold_train, fold_val, test_seqs,
             pretrain_meta, epochs: int, select_metric: str = "auc") -> None:
    variant = model_name if select_metric == "auc" else f"{model_name}_bsel"
    metrics_path, pred_path, error_path = cell_paths(variant, fold)
    if os.path.exists(metrics_path):
        print(f"[SKIP] {variant} fold{fold} already done ({metrics_path})")
        return

    seed = TRAIN_SEED_BASE + fold
    set_seed(seed)
    print(f"\n===== {variant} | fold {fold} | seed {seed} | epochs {epochs} | select={select_metric} =====")
    print(f"train shots={len(fold_train)} | val shots={len(fold_val)} | test shots={len(test_seqs)}")

    model, train_fn, predict_fn, make_loader = build_cell(
        model_name, fold_train, fold_val, test_seqs, pretrain_meta
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params = {n_trainable:,}")

    train_labels = [int(s["label"]) for s in fold_train]
    class_weights = compute_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr = MLP_LR if model_name == "mlp" else LR  # 各模型沿用自身脚本的默认学习率
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=WEIGHT_DECAY
    )

    train_loader = make_loader(fold_train, True)
    val_loader   = make_loader(fold_val, False)
    test_loader  = make_loader(test_seqs, False)

    best_score = -float("inf")
    best_threshold = 0.5
    best_state: Optional[dict] = None

    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer, criterion)
        val_loss, val_pred = predict_fn(model, val_loader, criterion)
        thr, _ = find_best_threshold(val_pred["label"].tolist(), val_pred["goal_prob"].tolist())
        vm = evaluate_prediction_df(val_pred, threshold=thr)
        print(f"[{variant} f{fold}] Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_auc={vm['roc_auc']:.4f} | val_brier={vm['brier']:.4f} | "
              f"val_f1={vm['f1']:.4f} | thr={thr:.2f}")

        score = -vm["brier"] if select_metric == "brier" else vm["roc_auc"]
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_threshold = thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_pred = predict_fn(model, val_loader, criterion)
    _, test_pred = predict_fn(model, test_loader, criterion)

    rows = []
    for split_name, pred_df in [("val", val_pred), ("test", test_pred)]:
        m = evaluate_prediction_df(pred_df, threshold=best_threshold)
        rows.append({
            "model": variant, "fold": fold, "seed": seed, "epochs": epochs,
            "select_metric": select_metric,
            "split": split_name, "n_shots": len(pred_df),
            "positive_rate": m["positive_rate"], "threshold": best_threshold,
            "f1": m["f1"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier"],
            "best_val_score": best_score,
        })
        print(f"[{variant} f{fold}] {split_name}: f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
              f"ll={m['log_loss']:.4f} brier={m['brier']:.4f}")

    test_pred["model"] = variant
    test_pred["fold"] = fold
    atomic_write_csv(test_pred, pred_path)
    atomic_write_csv(pd.DataFrame(rows), metrics_path)
    if os.path.exists(error_path):
        os.remove(error_path)
    print(f"[DONE] {variant} fold{fold} → {metrics_path}")

    del model, train_loader, val_loader, test_loader, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Aggregation: mean/SD/SE/95%CI + paired comparisons
# ---------------------------------------------------------------------------
def t_ppf_975(df_: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df_))
    except ImportError:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df_, 1.96)


def paired_t_pvalue(diffs: np.ndarray) -> float:
    try:
        from scipy import stats
        return float(stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)
    except ImportError:
        return float("nan")


def aggregate() -> None:
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "xg_kfold_*_fold*_metrics.csv")))
    if not files:
        print("No cell metrics found; nothing to aggregate.")
        return
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    atomic_write_csv(combined, COMBINED_METRICS)

    test_df = combined[combined["split"] == "test"]
    summary_rows = []
    print("\n================ K-FOLD SUMMARY (test set) ================")
    for model_name, g in test_df.groupby("model"):
        n = len(g)
        tcrit = t_ppf_975(n - 1) if n > 1 else float("nan")
        line = f"{model_name} (n_folds={n}): "
        row = {"kind": "model", "model": model_name, "n_folds": n}
        for metric in METRIC_COLS:
            vals = g[metric].to_numpy(dtype=float)
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_lo"] = mean - tcrit * se if n > 1 else float("nan")
            row[f"{metric}_ci95_hi"] = mean + tcrit * se if n > 1 else float("nan")
            line += f"{metric}={mean:.4f}±{se:.4f} "
        summary_rows.append(row)
        print(line)

    models_done = sorted(test_df["model"].unique())
    print("\n---------------- Paired comparisons (test) ----------------")
    for i in range(len(models_done)):
        for j in range(i + 1, len(models_done)):
            a, b = models_done[i], models_done[j]
            ga = test_df[test_df["model"] == a].set_index("fold").sort_index()
            gb = test_df[test_df["model"] == b].set_index("fold").sort_index()
            common = ga.index.intersection(gb.index)
            if len(common) < 2:
                continue
            row = {"kind": "paired_diff", "model": f"{b} - {a}", "n_folds": len(common)}
            tcrit = t_ppf_975(len(common) - 1)
            line = f"{b} - {a} (n={len(common)}): "
            for metric in METRIC_COLS:
                diffs = (gb.loc[common, metric].to_numpy(dtype=float)
                         - ga.loc[common, metric].to_numpy(dtype=float))
                mean, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
                se = sd / math.sqrt(len(common))
                pval = paired_t_pvalue(diffs)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_se"] = se
                row[f"{metric}_ci95_lo"] = mean - tcrit * se
                row[f"{metric}_ci95_hi"] = mean + tcrit * se
                row[f"{metric}_p_paired_t"] = pval
                line += f"Δ{metric}={mean:+.4f} (p={pval:.3f}) "
            summary_rows.append(row)
            print(line)

    atomic_write_csv(pd.DataFrame(summary_rows), SUMMARY_CSV)
    print(f"\nSaved: {COMBINED_METRICS}, {SUMMARY_CSV}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.aggregate_only:
        aggregate()
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise SystemExit(f"Unknown models: {invalid}. Choose from {ALL_MODELS}")

    folds = list(range(args.k)) if args.folds is None else [int(f) for f in args.folds.split(",")]

    print(f"DEVICE = {DEVICE}")
    print(f"MODELS = {models} | K = {args.k} | FOLDS = {folds} | EPOCHS = {args.epochs}")
    print(f"FOLD_SEED = {FOLD_SEED} | TRAIN_SEED = {TRAIN_SEED_BASE}+fold")

    pretrain_meta = maybe_load_pretrain_metadata()
    if pretrain_meta is not None:
        cat_cols = list(pretrain_meta["cat_cols"])
        cont_cols = list(pretrain_meta["cont_cols"])
    else:
        cat_cols = list(DEFAULT_CAT_COLS)
        cont_cols = list(DEFAULT_CONT_COLS)

    max_train = args.max_train_matches if args.max_train_matches is not None else MAX_TRAIN_MATCH_FILES
    max_test  = args.max_test_matches if args.max_test_matches is not None else MAX_TEST_MATCH_FILES
    use_cache = not args.no_seq_cache and args.max_train_matches is None

    train_all, test_seqs = load_all_sequences(cat_cols, cont_cols, max_train, max_test, use_cache)
    print(f"Total train-pool shots = {len(train_all)} | test shots = {len(test_seqs)}")

    assign = get_fold_assignments(train_all, args.k)

    for fold in folds:
        fold_train, fold_val = split_fold(train_all, assign, fold)
        for model_name in models:
            variant = model_name if args.select_metric == "auc" else f"{model_name}_bsel"
            metrics_path, _, error_path = cell_paths(variant, fold)
            if os.path.exists(metrics_path):
                print(f"[SKIP] {variant} fold{fold} already done")
                continue
            try:
                run_cell(model_name, fold, fold_train, fold_val, test_seqs,
                         pretrain_meta, args.epochs, args.select_metric)
            except Exception:
                err = traceback.format_exc()
                print(f"[ERROR] {variant} fold{fold}:\n{err}")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(err)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    aggregate()


if __name__ == "__main__":
    main()
