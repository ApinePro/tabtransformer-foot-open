"""
train_vaep_tabtransformer_catboost.py

对照实验：冻结预训练 TabEventEncoder 提取 embedding，然后用 CatBoost 做下游分类。

与其他 VAEP 脚本的对比：
  - train_vaep_tabtransformer.py    : 冻结 encoder + MLP 头（PyTorch）
  - train_vaep_tabtransformer_catboost.py : 冻结 encoder + CatBoost 头（本脚本）
  - train_vaep_catboost.py          : CatBoost（直接用 StatsBomb 原始特征，无 encoder）

内存策略（16GB 机器）：
  - 完全绕过 pkl 缓存，逐 parquet 文件流式处理
  - 每次只处理一个 match 文件（~3500 行），处理完立即释放
  - embedding 矩阵通过 np.memmap 直接写磁盘，不占 RAM
  - CatBoost Pool 量化后约 5GB，在 10GB 可用 RAM 内运行

流程：
  1. 加载预训练 TabEventEncoder，冻结参数
  2. 按 match_id 做 train/val 文件级别分割（复现 split_train_val_by_match 逻辑）
  3. 两阶段提取 embedding：
     a. 快速计数遍（仅读 type 列）→ 确定 N
     b. 处理遍（逐文件推理）→ 写 memmap
  4. 用 CatBoost 在 memmap embedding 上训练 score / concede 分类器
  5. 保存预测结果和评估指标

用法：
  python train_vaep_tabtransformer_catboost.py
  python train_vaep_tabtransformer_catboost.py --gpu
  python train_vaep_tabtransformer_catboost.py --eval-only
"""

import argparse
import gc
import hashlib
import os
import pickle
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    RAW_EVENT_COLUMNS,
    TabEventEncoder,
    set_seed,
)
from train_vaep_baseline import (
    CACHE_DIR,
    MAX_TEST_MATCH_FILES,
    MAX_TRAIN_MATCH_FILES,
    SEED,
    SKIP_TYPES,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    add_vaep_columns,
    binary_log_loss,
    build_actions_from_match,
    choose_prediction_output_path,
    find_best_threshold,
    list_event_files,
    safe_num,
    safe_text,
    save_predictions_compact,
    summarize_split,
)
from train_vaep_tabtransformer import (
    actions_to_arrays,
    load_pretrained_encoder,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

# ---------------------------------------------------------------------------
PRETRAIN_CHECKPOINT = os.path.join("checkpoints", "tabtransformer_row_mask_pretrain.pt")
SCORE_MODEL_PATH    = os.path.join("checkpoints", "vaep_tt_catboost_score.cbm")
CONCEDE_MODEL_PATH  = os.path.join("checkpoints", "vaep_tt_catboost_concede.cbm")
PREDICTIONS_OUTPUT  = "vaep_tabtransformer_catboost_predictions"
METRICS_OUTPUT      = "vaep_tabtransformer_catboost_metrics.csv"
EMBED_CACHE_DIR     = os.path.join(CACHE_DIR, "tt_embeddings")

ITERATIONS            = 1000
DEPTH                 = 6
LEARNING_RATE         = 0.1
L2_LEAF_REG           = 3.0
EARLY_STOPPING_ROUNDS = 50
EMBED_BATCH_SIZE      = 2048
LOOKAHEAD_K           = 10
PCA_DIM               = 128   # IncrementalPCA output dim; 0 = disabled (needs 25GB+ disk)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VAEP: frozen TabTransformer encoder + CatBoost head"
    )
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    p.add_argument("--score-model-path",    default=SCORE_MODEL_PATH)
    p.add_argument("--concede-model-path",  default=CONCEDE_MODEL_PATH)
    p.add_argument("--iterations",    type=int,   default=ITERATIONS)
    p.add_argument("--depth",          type=int,   default=DEPTH)
    p.add_argument("--lr",             type=float, default=LEARNING_RATE)
    p.add_argument("--l2",             type=float, default=L2_LEAF_REG)
    p.add_argument("--early-stopping", type=int,   default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--embed-batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--gpu", action="store_true", help="Use GPU for CatBoost training")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    p.add_argument("--prediction-splits", default="val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches",  type=int, default=None)
    p.add_argument("--pca-dim", type=int, default=PCA_DIM,
                   help="IncrementalPCA output dim (0 = disabled, needs ~25GB disk)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File-level train/val split (replicates split_train_val_by_match logic)
# ---------------------------------------------------------------------------
def _read_match_id(path: str) -> Optional[int]:
    """Read the match_id from the first row of a parquet file (cheap)."""
    try:
        df = pd.read_parquet(path, columns=["match_id"])
        if len(df) > 0 and "match_id" in df.columns:
            return int(df["match_id"].iloc[0])
    except Exception:
        pass
    return None


def split_files_train_val(
    file_list: List[Tuple[str, str]],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split file list into train/val by match_id.
    Reproduces split_train_val_by_match: sort unique match_ids, shuffle,
    take first n_val as validation.
    """
    # Build (match_id, path, source) list
    entries = []
    for path, src in tqdm(file_list, desc="Reading match_ids", leave=False):
        mid = _read_match_id(path)
        if mid is not None:
            entries.append((mid, path, src))
        else:
            entries.append((-1, path, src))  # fallback: keep in train

    # Sort by match_id (same as split_train_val_by_match uses sorted unique ids)
    match_ids = sorted({e[0] for e in entries})
    rng = random.Random(seed)
    rng.shuffle(match_ids)
    n_val = max(1, int(len(match_ids) * val_ratio))
    val_ids = set(match_ids[:n_val])

    train_files = [(p, s) for mid, p, s in entries if mid not in val_ids]
    val_files   = [(p, s) for mid, p, s in entries if mid in val_ids]
    return train_files, val_files


# ---------------------------------------------------------------------------
# Streaming embedding extraction
# ---------------------------------------------------------------------------
def _files_cache_key(file_list: List[Tuple[str, str]]) -> str:
    paths = sorted(p for p, _ in file_list)
    return hashlib.md5("|".join(paths).encode()).hexdigest()[:12]


def _encode_file(
    encoder: TabEventEncoder,
    path: str,
    src: str,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: dict,
    device: torch.device,
    batch_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list]]:
    """Read one parquet match file, encode with frozen encoder.
    Returns (emb[M, D], score[M], concede[M], mids, srcs, aidxs) or None.
    Each file is ~3500 rows → emb ≈ 12MB RAM, safe to hold temporarily.
    """
    try:
        try:
            df = pd.read_parquet(path, columns=[c for c in RAW_EVENT_COLUMNS if c != "id"])
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in RAW_EVENT_COLUMNS if c in df.columns]
            df = df[keep].copy()
    except Exception as e:
        print(f"  Skipping {path}: {e}")
        return None

    actions = build_actions_from_match(df, src, lookahead_k=LOOKAHEAD_K)
    del df
    if not actions:
        return None

    (
        cat_arr, cont_arr, score_arr, concede_arr,
        m_ids, s_names, a_idxs
    ) = actions_to_arrays(actions, cat_cols, cont_cols, cat_maps, desc="")
    del actions
    gc.collect()

    M = len(cat_arr)
    emb_chunks = []
    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            x_cat  = torch.from_numpy(cat_arr[start:end]).to(device)
            x_cont = torch.from_numpy(cont_arr[start:end]).to(device)
            emb_chunks.append(encoder(x_cat, x_cont).cpu().float().numpy())
    emb = np.concatenate(emb_chunks, axis=0) if len(emb_chunks) > 1 else emb_chunks[0]
    del cat_arr, cont_arr, emb_chunks
    gc.collect()

    return emb, score_arr.astype(np.int8), concede_arr.astype(np.int8), m_ids, s_names, a_idxs


def _count_actions_in_files(file_list: List[Tuple[str, str]]) -> int:
    """Fast count pass: read only 'type' column, count non-SKIP_TYPES rows."""
    total = 0
    for path, _ in tqdm(file_list, desc="Counting rows", leave=False):
        try:
            df = pd.read_parquet(path, columns=["type"])
            total += int((~df["type"].isin(SKIP_TYPES)).sum())
        except Exception:
            pass
    return total


def extract_embeddings_streaming(
    encoder: TabEventEncoder,
    file_list: List[Tuple[str, str]],
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: dict,
    device: torch.device,
    batch_size: int,
    name: str,
    pca_dim: int,
    use_cache: bool = True,
    fitted_pca=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list, object]:
    """
    逐文件流式提取 embedding，可选 IncrementalPCA 降维后写入 memmap。

    内存 / 磁盘占用（默认 pca_dim=128）:
        - 每次 RAM：单文件 embedding ~12MB，可忽略不计
        - 磁盘：5.7M × 128 × 4B ≈ 2.9GB（远低于原始的 20.8GB）
        - CatBoost Pool：~0.7GB，在 10GB 可用 RAM 内完全可行

    pca_dim=0 时禁用 PCA，直接写 output_dim 维（需要 ~21GB 磁盘）。

    返回: (embeddings_memmap, score_labels, concede_labels,
            match_ids, source_names, action_indices, fitted_pca_or_None)
    """
    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    key        = _files_cache_key(file_list) + (f"_pca{pca_dim}" if pca_dim > 0 else "")
    bin_path   = os.path.join(EMBED_CACHE_DIR, f"emb_{name}_{key}.bin")
    shape_path = os.path.join(EMBED_CACHE_DIR, f"emb_{name}_{key}_shape.npy")
    label_path = os.path.join(EMBED_CACHE_DIR, f"emb_{name}_{key}_labels.npz")
    pca_path   = os.path.join(EMBED_CACHE_DIR, f"emb_{name}_{key}_pca.pkl")

    if use_cache and os.path.exists(bin_path) and os.path.exists(shape_path) \
            and os.path.exists(label_path):
        print(f"Loading cached embeddings for '{name}' from {bin_path}")
        shape = tuple(np.load(shape_path).tolist())
        embeddings = np.memmap(bin_path, dtype="float32", mode="r", shape=shape)
        data = np.load(label_path, allow_pickle=True)
        if fitted_pca is None and pca_dim > 0 and os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                fitted_pca = pickle.load(f)
        return (
            embeddings,
            data["score_labels"],
            data["concede_labels"],
            data["match_ids"].tolist(),
            data["source_names"].tolist(),
            data["action_indices"].tolist(),
            fitted_pca,
        )

    # --- Pass 1: count total N ---
    print(f"[{name}] Counting actions across {len(file_list)} files...")
    N = _count_actions_in_files(file_list)
    print(f"[{name}] Total actions = {N:,}")

    encoder = encoder.to(device)
    encoder.eval()

    # --- Pass 2 (only for train): fit IncrementalPCA ---
    if pca_dim > 0 and fitted_pca is None:
        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components=pca_dim)
        print(f"[{name}] Fitting IncrementalPCA(n_components={pca_dim}) — pass 1/{2 if pca_dim > 0 else 1} ...")
        for path, src in tqdm(file_list, desc=f"[{name}] Fit PCA"):
            result = _encode_file(encoder, path, src, cat_cols, cont_cols, cat_maps,
                                  device, batch_size)
            if result is None:
                continue
            emb = result[0]  # [M, output_dim]
            # partial_fit in sub-batches (need batch >= n_components)
            for i in range(0, len(emb), batch_size):
                chunk = emb[i:i + batch_size]
                if len(chunk) >= pca_dim:
                    ipca.partial_fit(chunk)
            del result, emb
            gc.collect()
        fitted_pca = ipca
        with open(pca_path, "wb") as f:
            pickle.dump(fitted_pca, f)
        print(f"[{name}] Saved fitted PCA → {pca_path}")
        explained = float(np.sum(fitted_pca.explained_variance_ratio_)) * 100
        print(f"[{name}] PCA explained variance ({pca_dim} components): {explained:.1f}%")

    out_dim = pca_dim if pca_dim > 0 else encoder.output_dim
    disk_gb = N * out_dim * 4 / 1e9
    print(f"[{name}] Allocating memmap {bin_path}  ({N:,} × {out_dim}, {disk_gb:.1f} GB)")

    # --- Pass 3 (or Pass 2 if no PCA): encode + write memmap ---
    embeddings = np.memmap(bin_path, dtype="float32", mode="w+", shape=(N, out_dim))
    score_labels_arr   = np.empty(N, dtype=np.int8)
    concede_labels_arr = np.empty(N, dtype=np.int8)
    match_ids_list:    list = []
    source_names_list: list = []
    action_idxs_list:  list = []

    offset = 0
    pass_label = f"[{name}] Encode+Write" + (" (PCA transform)" if pca_dim > 0 else "")
    for path, src in tqdm(file_list, desc=pass_label):
        result = _encode_file(encoder, path, src, cat_cols, cont_cols, cat_maps,
                              device, batch_size)
        if result is None:
            continue
        emb, score, concede, m_ids, s_names, a_idxs = result
        if pca_dim > 0 and fitted_pca is not None:
            emb = fitted_pca.transform(emb).astype(np.float32)
        M = len(emb)
        embeddings[offset:offset + M] = emb
        score_labels_arr[offset:offset + M]   = score
        concede_labels_arr[offset:offset + M] = concede
        match_ids_list.extend(m_ids)
        source_names_list.extend(s_names)
        action_idxs_list.extend(a_idxs)
        offset += M
        del result, emb
        gc.collect()

    actual_N = offset
    if actual_N != N:
        print(f"[{name}] Warning: counted {N} but got {actual_N}. Truncating.")
        embeddings.flush()
        del embeddings
        src_mmap = np.memmap(bin_path, dtype="float32", mode="r", shape=(N, out_dim))
        trunc = np.array(src_mmap[:actual_N])
        del src_mmap
        embeddings = np.memmap(bin_path, dtype="float32", mode="w+", shape=(actual_N, out_dim))
        embeddings[:] = trunc
        del trunc
        score_labels_arr   = score_labels_arr[:actual_N]
        concede_labels_arr = concede_labels_arr[:actual_N]
        N = actual_N

    embeddings.flush()
    del embeddings

    np.save(shape_path, np.array([N, out_dim]))
    np.savez(
        label_path,
        score_labels   = score_labels_arr,
        concede_labels = concede_labels_arr,
        match_ids      = np.array(match_ids_list, dtype=np.int32),
        source_names   = np.array(source_names_list),
        action_indices = np.array(action_idxs_list, dtype=np.int32),
    )
    print(f"[{name}] Cached → {bin_path}  (shape={N}×{out_dim})")

    embeddings = np.memmap(bin_path, dtype="float32", mode="r", shape=(N, out_dim))
    return (
        embeddings,
        score_labels_arr,
        concede_labels_arr,
        match_ids_list,
        source_names_list,
        action_idxs_list,
        fitted_pca,
    )


# ---------------------------------------------------------------------------
# CatBoost training & prediction on embeddings
# ---------------------------------------------------------------------------
def train_catboost_on_embeddings(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
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
    print(f"\nTraining CatBoost for '{label}' on embeddings "
          f"(dim={train_emb.shape[1]}, n_train={len(train_emb)}, n_val={len(val_emb)}) ...")
    train_pool = Pool(train_emb, label=train_labels)
    val_pool   = Pool(val_emb,   label=val_labels)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def predict_on_embeddings(
    model: CatBoostClassifier,
    embeddings: np.ndarray,
    labels: np.ndarray,
    match_ids: list,
    source_names: list,
    action_indices: list,
    label_col: str,
    prob_col: str,
) -> pd.DataFrame:
    probs = model.predict_proba(embeddings)[:, 1]
    return pd.DataFrame({
        "match_id":     match_ids,
        "source_name":  source_names,
        "action_index": action_indices,
        label_col:      labels.astype(int),
        prob_col:       probs,
    })


# ---------------------------------------------------------------------------
def _build_pred_df(score_df: pd.DataFrame, concede_df: pd.DataFrame) -> pd.DataFrame:
    if len(score_df) == 0:
        return pd.DataFrame(columns=[
            "match_id", "source_name", "action_index",
            "score_label", "concede_label", "score_prob", "concede_prob",
        ])
    return score_df.merge(
        concede_df[["match_id", "action_index", "concede_label", "concede_prob"]],
        on=["match_id", "action_index"],
        how="inner",
    )


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

    os.makedirs("checkpoints", exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load pretrained encoder                                           #
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.pretrain_checkpoint):
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {args.pretrain_checkpoint}"
        )
    print(f"Loading pretrained encoder from {args.pretrain_checkpoint} ...")
    encoder, cat_cols, cont_cols, cat_maps, cat_dims, cont_mean_std = \
        load_pretrained_encoder(args.pretrain_checkpoint)
    print(f"Encoder output_dim = {encoder.output_dim}")

    # ------------------------------------------------------------------ #
    # 2. Build file lists (no action loading yet)                          #
    # ------------------------------------------------------------------ #
    max_train = args.max_train_matches or MAX_TRAIN_MATCH_FILES
    max_test  = args.max_test_matches  or MAX_TEST_MATCH_FILES

    all_train_files = list_event_files(TRAIN_ROOT, max_match_files=max_train)
    test_files      = list_event_files(TEST_ROOT,  max_match_files=max_test)

    print(f"Splitting {len(all_train_files)} train files into train/val ...")
    train_files, val_files = split_files_train_val(all_train_files, VAL_RATIO, SEED)
    print(f"train files={len(train_files)} | val files={len(val_files)} | test files={len(test_files)}")

    # ------------------------------------------------------------------ #
    # 3. Streaming embedding extraction                                    #
    # ------------------------------------------------------------------ #
    pca_dim = args.pca_dim
    print(f"PCA_DIM = {pca_dim}  {'(disabled — needs ~25 GB disk)' if pca_dim == 0 else f'(disk: ~{5744000 * pca_dim * 4 / 1e9:.1f} GB for train)'}")  # noqa

    def get_emb(name, files, fitted_pca=None):
        return extract_embeddings_streaming(
            encoder, files, cat_cols, cont_cols, cat_maps,
            device, args.embed_batch_size, name, pca_dim, use_cache, fitted_pca,
        )

    print("\n--- Extracting embeddings (streaming + IncrementalPCA) ---")
    (train_emb, train_score, train_concede,
     train_mids, train_srcs, train_aidxs, fitted_pca) = get_emb("train", train_files)
    gc.collect()

    (val_emb,   val_score,   val_concede,
     val_mids,  val_srcs,    val_aidxs, _) = get_emb("val",   val_files,  fitted_pca)
    gc.collect()

    (test_emb,  test_score,  test_concede,
     test_mids, test_srcs,   test_aidxs, _) = get_emb("test",  test_files, fitted_pca)
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
        print("\n--- Training CatBoost on embeddings ---")
        score_model = train_catboost_on_embeddings(
            train_emb, train_score, val_emb, val_score, "score", args
        )
        score_model.save_model(args.score_model_path)
        print(f"Saved score model → {args.score_model_path}")
        gc.collect()

        concede_model = train_catboost_on_embeddings(
            train_emb, train_concede, val_emb, val_concede, "concede", args
        )
        concede_model.save_model(args.concede_model_path)
        print(f"Saved concede model → {args.concede_model_path}")
        gc.collect()

    # ------------------------------------------------------------------ #
    # 5. Generate predictions                                              #
    # ------------------------------------------------------------------ #
    print("\n--- Generating predictions ---")

    def pred(emb, score_lbl, concede_lbl, mids, srcs, aidxs):
        s_df = predict_on_embeddings(
            score_model, emb, score_lbl, mids, srcs, aidxs,
            "score_label", "score_prob",
        )
        c_df = predict_on_embeddings(
            concede_model, emb, concede_lbl, mids, srcs, aidxs,
            "concede_label", "concede_prob",
        )
        return _build_pred_df(s_df, c_df)

    train_pred = (pred(train_emb, train_score, train_concede,
                       train_mids, train_srcs, train_aidxs)
                  if "train" in valid_splits else pd.DataFrame())
    val_pred   = pred(val_emb,  val_score,  val_concede,
                      val_mids, val_srcs,   val_aidxs)
    test_pred  = pred(test_emb, test_score, test_concede,
                      test_mids, test_srcs, test_aidxs)

    _export(args, valid_splits, train_pred, val_pred, test_pred)


if __name__ == "__main__":
    main()

