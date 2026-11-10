"""
train_xg_gru.py

对照实验：GRU 作为下游序列模型的 xG 任务，与现有两个模型构成三方对比：
  1. MLP baseline（已有: train_xg_baseline_mlp.py，仅终末射门事件，无序列）
  2. basic + GRU      : 随机初始化 per-column embedding（TabMLPBaseline 同款事件编码）+ GRU
  3. pretrained + GRU : 预训练 TabEventEncoder（911 维事件 embedding）+ GRU

除序列建模器（GRU 替换 Transformer encoder）外，其余配置与
train_xg_tabtransformer.py 完全一致：
  - 同一数据管线 / 序列切分 / 防泄漏策略（shot_end_location 移除）
  - 同一 match-level train/val 划分（SEED=42, VAL_RATIO=0.1）与 test 赛事
  - MAX_SEQ_LEN=48, BATCH_SIZE=8, EPOCHS=5, LR=1e-4, WEIGHT_DECAY=1e-4
  - HIDDEN_DIM=128, DROPOUT=0.1, GRU 层数 = 2（对齐 NUM_SEQUENCE_LAYERS）
  - 同一类别权重方案 / val-AUC 选 checkpoint / val 上调分类阈值
  - 同一指标导出格式（overall + 分赛事）

用法：
  python train_xg_gru.py --encoder basic
  python train_xg_gru.py --encoder pretrained                    # 微调 encoder（对齐 README 默认 run）
  python train_xg_gru.py --encoder pretrained --freeze-encoder   # 冻结 encoder（对齐实验矩阵 ft-frozen run）
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    BATCH_SIZE,
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    DEVICE,
    DROPOUT,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    PRETRAIN_CHECKPOINT,
    SEED,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    WEIGHT_DECAY,
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
    split_train_val_by_match,
    train_one_epoch,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

GRU_LAYERS = 2  # 对齐 SequenceXGModel 的 NUM_SEQUENCE_LAYERS


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xG downstream with GRU sequence model")
    p.add_argument("--encoder", choices=["basic", "pretrained"], required=True,
                   help="basic: random per-column embeddings; pretrained: frozen/fine-tuned TabEventEncoder")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze the pretrained event encoder (only valid with --encoder pretrained)")
    p.add_argument("--max-train-matches", type=int, default=None, help="Smoke test: limit train files")
    p.add_argument("--max-test-matches", type=int, default=None, help="Smoke test: limit test files")
    return p.parse_args()


def variant_tag(args: argparse.Namespace) -> str:
    tag = f"xg_gru_{args.encoder}"
    if args.encoder == "pretrained" and args.freeze_encoder:
        tag += "_frozen"
    if EPOCHS != 5:  # 非默认轮数（如 XG_EPOCHS=20 对齐 G2F）输出到独立文件
        tag += f"_ep{EPOCHS}"
    return tag


# ---------------------------------------------------------------------------
class BasicEventEncoder(nn.Module):
    """TabMLPBaseline 同款逐事件编码：per-column embedding + z-norm 连续特征拼接。"""

    def __init__(self, cat_dims: List[int], num_cont: int, cont_mean_std: np.ndarray):
        super().__init__()
        emb_dims = [max(4, min(32, int(round(1.6 * (dim ** 0.56))))) for dim in cat_dims]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, emb_dim) for dim, emb_dim in zip(cat_dims, emb_dims)]
        )
        mean_std = torch.tensor(cont_mean_std, dtype=torch.float32)
        self.register_buffer("cont_mean", mean_std[:, 0])
        self.register_buffer("cont_std", mean_std[:, 1])
        self.output_dim = int(sum(emb_dims) + num_cont)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_categ[:, i].clamp_min(0)) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embs, dim=-1)
        x_cont_norm = (x_cont - self.cont_mean) / self.cont_std
        return torch.cat([x_cat, x_cont_norm], dim=-1)


class GRUXGModel(nn.Module):
    """事件编码器（basic 或 pretrained）+ 2 层 GRU + 末有效时刻隐状态分类。"""

    def __init__(self, event_encoder: nn.Module, encoder_output_dim: int):
        super().__init__()
        self.event_encoder = event_encoder
        self.event_proj = nn.Sequential(
            nn.Linear(encoder_output_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=GRU_LAYERS,
            batch_first=True,
            dropout=DROPOUT if GRU_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 2))

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_cat = x_categ.shape
        _, _, num_cont = x_cont.shape

        flat_categ = x_categ.view(batch_size * seq_len, num_cat)
        flat_cont = x_cont.view(batch_size * seq_len, num_cont)
        event_features = self.event_encoder(flat_categ, flat_cont)
        event_tokens = self.event_proj(event_features).view(batch_size, seq_len, -1)

        # 序列右侧补零，pack 后 GRU 只处理有效时刻；h_n[-1] 即末有效事件（射门）后的隐状态
        lengths = valid_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(event_tokens, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.head(h_n[-1])


# ---------------------------------------------------------------------------
def load_pretrained_backbone(encoder: TabEventEncoder, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = encoder.backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"已加载预训练 event encoder: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()
    if args.freeze_encoder and args.encoder != "pretrained":
        raise SystemExit("--freeze-encoder 仅对 --encoder pretrained 有效")

    tag = variant_tag(args)
    checkpoint_path = os.path.join("checkpoints", f"{tag}_best.pt")
    metrics_output = f"{tag}_metrics.csv"
    predictions_output = f"{tag}_predictions.csv"

    set_seed(SEED)
    print(f"DEVICE = {DEVICE}")
    print(f"VARIANT = {tag} (encoder={args.encoder}, freeze={args.freeze_encoder}, gru_layers={GRU_LAYERS})")

    # ---- 特征定义与编码映射：与原实验同源 ----
    pretrain_meta = maybe_load_pretrain_metadata()
    if args.encoder == "pretrained":
        if pretrain_meta is None:
            raise FileNotFoundError(f"Pretrain checkpoint required: {PRETRAIN_CHECKPOINT}")
        cat_cols = list(pretrain_meta["cat_cols"])
        cont_cols = list(pretrain_meta["cont_cols"])
        cat_maps = pretrain_meta["cat_maps"]
        cat_dims = list(pretrain_meta["cat_dims"])
        cont_mean_std = np.asarray(pretrain_meta["cont_mean_std"], dtype=np.float32)
        print("使用预训练阶段保存的特征定义与编码映射。")
    else:
        if pretrain_meta is not None:
            cat_cols = list(pretrain_meta["cat_cols"])
            cont_cols = list(pretrain_meta["cont_cols"])
        else:
            cat_cols = list(DEFAULT_CAT_COLS)
            cont_cols = list(DEFAULT_CONT_COLS)
        cat_maps = None
        cat_dims = None
        cont_mean_std = None

    # ---- 数据加载与划分：与原实验完全一致 ----
    train_sequences_all = load_sequences_from_root(
        TRAIN_ROOT, cat_cols, cont_cols,
        max_match_files=args.max_train_matches if args.max_train_matches is not None else MAX_TRAIN_MATCH_FILES,
    )
    test_sequences = load_sequences_from_root(
        TEST_ROOT, cat_cols, cont_cols,
        max_match_files=args.max_test_matches if args.max_test_matches is not None else MAX_TEST_MATCH_FILES,
    )
    train_sequences, val_sequences = split_train_val_by_match(train_sequences_all, val_ratio=VAL_RATIO, seed=SEED)

    if cat_maps is None or cat_dims is None:
        cat_maps, cat_dims = build_cat_maps_from_sequences(train_sequences, cat_cols)
        cont_mean_std = compute_cont_stats(train_sequences, cont_cols)

    print(f"训练 shots: {len(train_sequences)} | 验证 shots: {len(val_sequences)} | 测试 shots: {len(test_sequences)}")
    train_labels = [int(seq["label"]) for seq in train_sequences]
    print(f"训练集进球率 = {np.mean(train_labels):.4f}")

    train_loader = DataLoader(
        ShotSequenceDataset(train_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        ShotSequenceDataset(val_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        ShotSequenceDataset(test_sequences, cat_cols, cont_cols, cat_maps),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # ---- 模型 ----
    if args.encoder == "pretrained":
        event_encoder = TabEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
        load_pretrained_backbone(event_encoder, PRETRAIN_CHECKPOINT)
        model = GRUXGModel(event_encoder, event_encoder.output_dim).to(DEVICE)
        if args.freeze_encoder:
            for param in model.event_encoder.parameters():
                param.requires_grad = False
            print("已冻结预训练 event encoder。")
    else:
        event_encoder = BasicEventEncoder(cat_dims, len(cont_cols), cont_mean_std)
        model = GRUXGModel(event_encoder, event_encoder.output_dim).to(DEVICE)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params = {n_trainable:,}")

    class_weights = compute_class_weights(train_labels)
    print(f"Class weights = {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ---- 训练循环：与原实验相同（val AUC 选 checkpoint，val 上调阈值）----
    os.makedirs("checkpoints", exist_ok=True)
    best_val_auc = -1.0
    best_threshold = 0.5

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_pred_df = predict_model(model, val_loader, criterion)
        current_threshold, _ = find_best_threshold(
            val_pred_df["label"].tolist(), val_pred_df["goal_prob"].tolist()
        )
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
                    "variant": tag,
                },
                checkpoint_path,
            )
            print(f"已保存最佳模型: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        best_ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_threshold = float(best_ckpt.get("best_threshold", 0.5))

    # ---- 测试评估：overall + 分赛事，与原指标格式一致 ----
    test_loss, test_pred_df = predict_model(model, test_loader, criterion)
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

    pd.DataFrame(rows).to_csv(metrics_output, index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(predictions_output, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics     : {metrics_output}")
    print(f"Saved predictions : {predictions_output}")


if __name__ == "__main__":
    main()
