"""
train_vaep_tabtransformer_catboost_full.py

对照实验：冻结 TabEventEncoder 全量 911 维 embedding（不做 PCA）+ CatBoost。

与 train_vaep_tabtransformer_catboost.py（PCA-128 版）的唯一区别是特征维度：
  - 本脚本   : encoder 原始输出 911 维
  - PCA-128 : IncrementalPCA 压缩到 128 维

超参数与 PCA-128 版完全一致（常量直接 import，保证不漂移），
仅特征维度不同，用于隔离“PCA 压缩损失”的对比。

16GB RAM 机器上的内存策略：
  - 911 维 float32 训练矩阵 ~21GB，无法直接构建 in-RAM CatBoost Pool
  - 改用 catboost.utils.quantize 从 TSV 文件流式量化（逐块读取+分箱），
    量化后 Pool ≈ N × 911 × 1B ≈ 5GB，可在 ~10GB 可用 RAM 内训练
  - score / concede 两个任务共享同一份 TSV 与同一份 quantization borders
  - 预测阶段直接对 memmap 分块 predict_proba，无需构建 Pool

磁盘占用（.vaep_cache/tt_embeddings/，可随时删除）：
  - 911 维 memmap：train ~21GB + val ~2.1GB + test ~1.9GB
  - 临时 TSV：train ~38GB + val ~3.8GB（训练完成后自动删除，--keep-tsv 保留）

用法：
  python train_vaep_tabtransformer_catboost_full.py
  python train_vaep_tabtransformer_catboost_full.py --eval-only
"""

import argparse
import gc
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

try:
    from catboost.utils import quantize as catboost_streaming_quantize
except ImportError:
    catboost_streaming_quantize = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import set_seed
from train_vaep_baseline import (
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    SEED,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    add_vaep_columns,
    binary_log_loss,
    choose_prediction_output_path,
    find_best_threshold,
    list_event_files,
    save_predictions_compact,
    summarize_split,
)
from train_vaep_tabtransformer import load_pretrained_encoder
from train_vaep_tabtransformer_catboost import (
    DEPTH,
    EARLY_STOPPING_ROUNDS,
    EMBED_BATCH_SIZE,
    EMBED_CACHE_DIR,
    ITERATIONS,
    L2_LEAF_REG,
    LEARNING_RATE,
    PRETRAIN_CHECKPOINT,
    _build_pred_df,
    _files_cache_key,
    extract_embeddings_streaming,
    split_files_train_val,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

# ---------------------------------------------------------------------------
SCORE_MODEL_PATH    = os.path.join("checkpoints", "vaep_tt_catboost_full_score.cbm")
CONCEDE_MODEL_PATH  = os.path.join("checkpoints", "vaep_tt_catboost_full_concede.cbm")
PREDICTIONS_OUTPUT  = "vaep_tabtransformer_catboost_full_predictions"
METRICS_OUTPUT      = "vaep_tabtransformer_catboost_full_metrics.csv"

TSV_CHUNK_ROWS     = 100_000
PREDICT_CHUNK_ROWS = 100_000


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VAEP: frozen TabTransformer encoder (full 911-dim, no PCA) + CatBoost"
    )
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    p.add_argument("--score-model-path",    default=SCORE_MODEL_PATH)
    p.add_argument("--concede-model-path",  default=CONCEDE_MODEL_PATH)
    p.add_argument("--iterations",     type=int,   default=ITERATIONS)
    p.add_argument("--depth",          type=int,   default=DEPTH)
    p.add_argument("--lr",             type=float, default=LEARNING_RATE)
    p.add_argument("--l2",             type=float, default=L2_LEAF_REG)
    p.add_argument("--early-stopping", type=int,   default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--embed-batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU for CatBoost (6GB VRAM is tight for 911-dim; CPU recommended)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    p.add_argument("--keep-tsv", action="store_true",
                   help="Keep intermediate TSV files after training (~42GB)")
    p.add_argument("--prediction-splits", default="val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches",  type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# TSV + column-description helpers for streaming quantization
# ---------------------------------------------------------------------------
def write_split_tsv(
    emb: np.ndarray,
    score_labels: np.ndarray,
    concede_labels: np.ndarray,
    tsv_path: str,
    chunk_rows: int = TSV_CHUNK_ROWS,
) -> None:
    """列布局: col0=score_label, col1=concede_label, col2.. = 911 维特征。"""
    if os.path.exists(tsv_path):
        print(f"TSV exists, reusing: {tsv_path}")
        return
    n = len(emb)
    tmp_path = tsv_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    est_gb = n * (emb.shape[1] * 8 + 4) / 1e9
    print(f"Writing TSV {tsv_path}  ({n:,} rows, est ~{est_gb:.1f} GB)")
    for start in tqdm(range(0, n, chunk_rows), desc=f"Write {os.path.basename(tsv_path)}"):
        end = min(start + chunk_rows, n)
        df = pd.DataFrame(np.asarray(emb[start:end]))
        df.insert(0, "__concede", np.asarray(concede_labels[start:end], dtype=np.int8))
        df.insert(0, "__score",   np.asarray(score_labels[start:end], dtype=np.int8))
        df.to_csv(tmp_path, sep="\t", header=False, index=False,
                  float_format="%.6g", mode="a")
        del df
    os.replace(tmp_path, tsv_path)


def write_cd_file(cd_path: str, task: str) -> None:
    """score: col0=Label col1=Auxiliary；concede: 反之。未列出的列默认为 Num。"""
    if task == "score":
        lines = ["0\tLabel", "1\tAuxiliary"]
    else:
        lines = ["0\tAuxiliary", "1\tLabel"]
    with open(cd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def quantize_from_tsv(
    tsv_path: str,
    cd_path: str,
    input_borders: Optional[str] = None,
) -> Pool:
    """catboost.utils.quantize 逐块读取 TSV 并量化，峰值 RAM ≈ 量化后 Pool 大小。"""
    kwargs = {"column_description": cd_path}
    if input_borders is not None:
        kwargs["input_borders"] = input_borders
    try:
        pool = catboost_streaming_quantize(tsv_path, **kwargs)
    except TypeError:
        # 旧版 catboost 不支持 input_borders 透传；同一 TSV 上重算 borders 结果一致
        kwargs.pop("input_borders", None)
        pool = catboost_streaming_quantize(tsv_path, **kwargs)
    print(f"  quantized pool: {pool.num_row():,} rows x {pool.num_col()} features")
    return pool


# ---------------------------------------------------------------------------
def train_catboost_on_pool(
    train_pool: Pool,
    val_pool: Pool,
    label: str,
    args: argparse.Namespace,
) -> CatBoostClassifier:
    task_type = "GPU" if args.gpu else "CPU"
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        task_type=task_type,
        early_stopping_rounds=args.early_stopping,
        verbose=100,
        auto_class_weights="Balanced",
    )
    print(f"\nTraining CatBoost for '{label}' on full-dim quantized pool ...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def predict_chunked(
    model: CatBoostClassifier,
    emb: np.ndarray,
    chunk_rows: int = PREDICT_CHUNK_ROWS,
    desc: str = "predict",
) -> np.ndarray:
    n = len(emb)
    probs = np.empty(n, dtype=np.float64)
    for start in tqdm(range(0, n, chunk_rows), desc=desc, leave=False):
        end = min(start + chunk_rows, n)
        probs[start:end] = model.predict_proba(np.asarray(emb[start:end]))[:, 1]
    return probs


def make_pred_df(
    score_model: CatBoostClassifier,
    concede_model: CatBoostClassifier,
    emb: np.ndarray,
    score_labels: np.ndarray,
    concede_labels: np.ndarray,
    match_ids: list,
    source_names: list,
    action_indices: list,
    name: str,
) -> pd.DataFrame:
    s_prob = predict_chunked(score_model,   emb, desc=f"[{name}] score")
    c_prob = predict_chunked(concede_model, emb, desc=f"[{name}] concede")
    s_df = pd.DataFrame({
        "match_id":      match_ids,
        "source_name":   source_names,
        "action_index":  action_indices,
        "score_label":   np.asarray(score_labels).astype(int),
        "score_prob":    s_prob,
    })
    c_df = pd.DataFrame({
        "match_id":      match_ids,
        "action_index":  action_indices,
        "concede_label": np.asarray(concede_labels).astype(int),
        "concede_prob":  c_prob,
    })
    return _build_pred_df(s_df, c_df)


# ---------------------------------------------------------------------------
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

    for df, name in [(train_pred, "train"), (val_pred, "val"), (test_pred, "test")]:
        if len(df) > 0:
            df["split"] = name

    pred_map = {"train": train_pred, "val": val_pred, "test": test_pred}
    parts = [pred_map[s] for s in valid_splits if len(pred_map.get(s, [])) > 0]
    if not parts:
        print("No predictions to export.")
        return

    all_pred = pd.concat(parts, ignore_index=True)
    all_pred = add_vaep_columns(all_pred)

    if not args.include_source_name and "source_name" in all_pred.columns:
        all_pred = all_pred.drop(columns=["source_name"])

    output_path = choose_prediction_output_path(PREDICTIONS_OUTPUT, args.prediction_format)
    save_predictions_compact(all_pred, output_path, args.prediction_format)

    metrics_rows = []
    for split_name, df_split in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        if len(df_split) > 0:
            metrics_rows.extend(summarize_split(split_name, df_split, score_thr, concede_thr))
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\nSaved predictions : {output_path}")
    print(f"Saved metrics     : {METRICS_OUTPUT}")
    print(f"score_thr={score_thr:.2f} | concede_thr={concede_thr:.2f}")

    if len(val_pred) > 0:
        s_ll = binary_log_loss(val_pred["score_label"].tolist(), val_pred["score_prob"].tolist())
        c_ll = binary_log_loss(val_pred["concede_label"].tolist(), val_pred["concede_prob"].tolist())
        print(f"Val score_log_loss={s_ll:.4f} | concede_log_loss={c_ll:.4f}")


# ---------------------------------------------------------------------------
def main():
    args      = parse_args()
    set_seed(SEED)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = not args.no_cache

    print(f"DEVICE = {device}")
    if device.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits    = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    if not args.eval_only and catboost_streaming_quantize is None:
        raise RuntimeError(
            "catboost.utils.quantize is unavailable in this catboost version. "
            "Streaming quantization is required for full 911-dim training on 16GB RAM. "
            "Upgrade with: pip install -U catboost"
        )

    os.makedirs("checkpoints", exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load pretrained encoder                                           #
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.pretrain_checkpoint):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_checkpoint}")
    print(f"Loading pretrained encoder from {args.pretrain_checkpoint} ...")
    encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std = \
        load_pretrained_encoder(args.pretrain_checkpoint)
    print(f"Encoder output_dim = {encoder.output_dim} (no PCA)")

    # ------------------------------------------------------------------ #
    # 2. File lists + split (identical to PCA-128 run: same seed/ratio)    #
    # ------------------------------------------------------------------ #
    max_train = args.max_train_matches or MAX_TRAIN_MATCH_FILES
    max_test  = args.max_test_matches  or MAX_TEST_MATCH_FILES

    all_train_files = list_event_files(TRAIN_ROOT, max_match_files=max_train)
    test_files      = list_event_files(TEST_ROOT,  max_match_files=max_test)

    print(f"Splitting {len(all_train_files)} train files into train/val ...")
    train_files, val_files = split_files_train_val(all_train_files, VAL_RATIO, SEED)
    print(f"train files={len(train_files)} | val files={len(val_files)} | test files={len(test_files)}")

    # ------------------------------------------------------------------ #
    # 3. Full-dim (911) embedding memmaps — shared cache with pca_dim=0    #
    # ------------------------------------------------------------------ #
    def get_emb(name, files):
        return extract_embeddings_streaming(
            encoder, files, cat_cols, cont_cols, cat_maps,
            device, args.embed_batch_size, name, 0, use_cache, None,
        )

    print("\n--- Extracting full-dim embeddings (streaming, no PCA) ---")
    (train_emb, train_score, train_concede,
     train_mids, train_srcs, train_aidxs, _) = get_emb("train", train_files)
    gc.collect()

    (val_emb, val_score, val_concede,
     val_mids, val_srcs, val_aidxs, _) = get_emb("val", val_files)
    gc.collect()

    (test_emb, test_score, test_concede,
     test_mids, test_srcs, test_aidxs, _) = get_emb("test", test_files)
    gc.collect()

    # ------------------------------------------------------------------ #
    # 4. Train or load CatBoost models                                     #
    # ------------------------------------------------------------------ #
    if args.eval_only:
        print("\n--- Eval-only: loading saved models ---")
        score_model   = CatBoostClassifier()
        score_model.load_model(args.score_model_path)
        concede_model = CatBoostClassifier()
        concede_model.load_model(args.concede_model_path)
    else:
        train_key = _files_cache_key(train_files)
        val_key   = _files_cache_key(val_files)
        train_tsv = os.path.join(EMBED_CACHE_DIR, f"full911_train_{train_key}.tsv")
        val_tsv   = os.path.join(EMBED_CACHE_DIR, f"full911_val_{val_key}.tsv")
        borders_path = os.path.join(EMBED_CACHE_DIR, f"full911_borders_{train_key}.tsv")
        cd_score   = os.path.join(EMBED_CACHE_DIR, "full911_cd_score.txt")
        cd_concede = os.path.join(EMBED_CACHE_DIR, "full911_cd_concede.txt")

        print("\n--- Writing TSVs for streaming quantization ---")
        write_split_tsv(train_emb, train_score, train_concede, train_tsv)
        write_split_tsv(val_emb,   val_score,   val_concede,   val_tsv)
        write_cd_file(cd_score, "score")
        write_cd_file(cd_concede, "concede")

        # ---- score task ----
        print("\n--- Quantizing pools (score) ---")
        train_pool = quantize_from_tsv(train_tsv, cd_score)
        train_pool.save_quantization_borders(borders_path)
        print(f"Saved quantization borders → {borders_path}")
        val_pool = quantize_from_tsv(val_tsv, cd_score, input_borders=borders_path)

        score_model = train_catboost_on_pool(train_pool, val_pool, "score", args)
        score_model.save_model(args.score_model_path)
        print(f"Saved score model → {args.score_model_path}")
        del train_pool, val_pool
        gc.collect()

        # ---- concede task (same features, same borders) ----
        print("\n--- Quantizing pools (concede) ---")
        train_pool = quantize_from_tsv(train_tsv, cd_concede, input_borders=borders_path)
        val_pool   = quantize_from_tsv(val_tsv,   cd_concede, input_borders=borders_path)

        concede_model = train_catboost_on_pool(train_pool, val_pool, "concede", args)
        concede_model.save_model(args.concede_model_path)
        print(f"Saved concede model → {args.concede_model_path}")
        del train_pool, val_pool
        gc.collect()

        if not args.keep_tsv:
            for path in (train_tsv, val_tsv):
                try:
                    os.remove(path)
                    print(f"Removed intermediate TSV: {path}")
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # 5. Chunked predictions from memmaps                                  #
    # ------------------------------------------------------------------ #
    print("\n--- Generating predictions ---")
    train_pred = (make_pred_df(score_model, concede_model, train_emb,
                               train_score, train_concede,
                               train_mids, train_srcs, train_aidxs, "train")
                  if "train" in valid_splits else pd.DataFrame())
    val_pred   = make_pred_df(score_model, concede_model, val_emb,
                              val_score, val_concede,
                              val_mids, val_srcs, val_aidxs, "val")
    test_pred  = make_pred_df(score_model, concede_model, test_emb,
                              test_score, test_concede,
                              test_mids, test_srcs, test_aidxs, "test")

    _export(args, valid_splits, train_pred, val_pred, test_pred)


if __name__ == "__main__":
    main()
