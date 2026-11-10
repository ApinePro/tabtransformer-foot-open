"""
train_vaep_catboost.py

用 CatBoost 做 VAEP score/concede 双任务预测（各训一个独立分类器）。
对标 Decroos 2019 原论文的方法，但输入是 StatsBomb 原始特征（比 SPADL 更丰富）。

与其他 VAEP 脚本的关系：
  - train_vaep_baseline.py  : 自训 embedding MLP
  - train_vaep_tabtransformer.py : 冻结预训练 encoder + MLP
  - train_vaep_catboost.py  : CatBoost（本脚本，无神经网络）

特点：
  - 类别列直接传给 CatBoost 的 cat_features，无需 embedding
  - 连续列直接使用原始值，CatBoost 内部处理缺失值
  - 与原论文一致：score / concede 各训独立模型
  - 复用 train_vaep_baseline 的数据缓存（.vaep_cache/）

用法：
  python train_vaep_catboost.py
  python train_vaep_catboost.py --eval-only
  python train_vaep_catboost.py --iterations 2000 --depth 8
"""

import argparse
import gc
import hashlib
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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
    build_actions_from_match,
    choose_prediction_output_path,
    find_best_threshold,
    get_cache_key,
    list_event_files,
    safe_num,
    safe_text,
    save_predictions_compact,
    summarize_split,
)
from train_xg_tabtransformer import (
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    set_seed,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
SCORE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vaep_catboost_score.cbm")
CONCEDE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vaep_catboost_concede.cbm")
PREDICTIONS_OUTPUT = "vaep_catboost_predictions"
METRICS_OUTPUT = "vaep_catboost_metrics.csv"

# CatBoost 默认超参数（对齐原论文精神）
ITERATIONS = 1000
DEPTH = 6
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
EARLY_STOPPING_ROUNDS = 50


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAEP with CatBoost")
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--l2", type=float, default=L2_LEAF_REG)
    p.add_argument("--early-stopping", type=int, default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--prediction-splits", default="val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches", type=int, default=None)
    p.add_argument("--gpu", action="store_true", help="Use GPU for CatBoost training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 直接从 parquet 文件流式构建 feature DataFrame
# 避免加载 5M+ action dict list（会消耗 10-15GB RAM）
# ---------------------------------------------------------------------------

def _split_files_train_val(
    files: List, val_ratio: float, seed: int
) -> tuple:
    """按文件（比赛）级别分 train/val，与 split_train_val_by_match 逻辑一致。"""
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]  # train, val


def build_feature_df_streaming(
    files: List,
    lookahead_k: int = 10,
    desc: str = "Building",
) -> pd.DataFrame:
    """逐比赛读取 parquet → 提取 action → 即时释放 dict，累积成 DataFrame。
    使用 sys.intern() 对 cat 字符串去重，大幅降低内存占用。
    """
    cat_cols_data  = {col: [] for col in DEFAULT_CAT_COLS}
    cont_cols_data = {col: [] for col in DEFAULT_CONT_COLS}
    score_labels:   List[int] = []
    concede_labels: List[int] = []
    match_ids:      List[int] = []
    source_names:   List[str] = []
    action_indices: List[int] = []

    for path, source_name in tqdm(files, desc=desc):
        try:
            df_raw = pd.read_parquet(path)
        except Exception as exc:
            print(f"Warning: skip {path} ({exc})")
            continue

        actions = build_actions_from_match(df_raw, source_name, lookahead_k)
        del df_raw

        for a in actions:
            feat = a["features"]
            for col in DEFAULT_CAT_COLS:
                cat_cols_data[col].append(sys.intern(safe_text(feat.get(col))))
            for col in DEFAULT_CONT_COLS:
                cont_cols_data[col].append(safe_num(feat.get(col)))
            score_labels.append(int(a["score_label"]))
            concede_labels.append(int(a["concede_label"]))
            match_ids.append(int(a["match_id"]))
            source_names.append(sys.intern(source_name))
            action_indices.append(int(a["action_index"]))
        del actions

    if not score_labels:
        return pd.DataFrame(columns=DEFAULT_CAT_COLS + DEFAULT_CONT_COLS +
                            ["score_label", "concede_label", "match_id", "source_name", "action_index"])

    data: Dict = {}
    for col in DEFAULT_CAT_COLS:
        data[col] = cat_cols_data[col]
    for col in DEFAULT_CONT_COLS:
        data[col] = np.array(cont_cols_data[col], dtype=np.float32)
    data["score_label"]   = np.array(score_labels,   dtype=np.int8)
    data["concede_label"] = np.array(concede_labels, dtype=np.int8)
    data["match_id"]      = np.array(match_ids,      dtype=np.int32)
    data["source_name"]   = source_names
    data["action_index"]  = np.array(action_indices, dtype=np.int32)
    return pd.DataFrame(data)


def load_or_build_feature_df(
    name: str,
    files: List,
    lookahead_k: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """缓存版：结果保存为 parquet（比 pkl 小很多），避免重复处理。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # cache key 用文件列表内容的 hash
    key_src = "|".join(sorted(p for p, _ in files)) + f"_k{lookahead_k}"
    key = hashlib.md5(key_src.encode()).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"cb_{name}_{key}.parquet")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached {name} feature df from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Building {name} feature df from {len(files)} match files...")
    df = build_feature_df_streaming(files, lookahead_k=lookahead_k, desc=f"  {name}")

    if use_cache and len(df) > 0:
        print(f"Caching {name} feature df → {cache_path}")
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            print(f"Warning: could not cache {name} ({exc})")
    return df


def make_pool(df: pd.DataFrame, label_col: str) -> Pool:
    X = df[DEFAULT_CAT_COLS + DEFAULT_CONT_COLS]
    y = df[label_col].values
    cat_feature_indices = list(range(len(DEFAULT_CAT_COLS)))
    return Pool(X, label=y, cat_features=cat_feature_indices)


# ---------------------------------------------------------------------------
# Train / predict helpers
# ---------------------------------------------------------------------------
def train_catboost(
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
    print(f"\nTraining CatBoost for {label} ...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def predict_with_model(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    label_col: str,
    prob_col: str,
) -> pd.DataFrame:
    X = df[DEFAULT_CAT_COLS + DEFAULT_CONT_COLS]
    probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "match_id": df["match_id"].values,
        "source_name": df["source_name"].values,
        "action_index": df["action_index"].values,
        label_col: df[label_col].values,
        prob_col: probs,
    })


# ---------------------------------------------------------------------------
def _export(args, valid_splits, train_df, val_df, test_df):
    # Merge score + concede predictions
    def merge_preds(score_df, concede_df):
        if len(score_df) == 0:
            return pd.DataFrame(columns=["match_id", "source_name", "action_index",
                                         "score_label", "concede_label",
                                         "score_prob", "concede_prob"])
        return score_df.merge(
            concede_df[["match_id", "action_index", "concede_label", "concede_prob"]],
            on=["match_id", "action_index"],
            how="inner",
        )

    train_pred = merge_preds(*train_df) if train_df else (pd.DataFrame(), pd.DataFrame())
    val_pred   = merge_preds(*val_df)
    test_pred  = merge_preds(*test_df)

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

    parts = []
    pred_map = {"train": train_pred, "val": val_pred, "test": test_pred}
    for s in valid_splits:
        df = pred_map.get(s, pd.DataFrame())
        if len(df) > 0:
            parts.append(df)

    if not parts:
        print("No predictions to export.")
        return

    all_pred = pd.concat(parts, ignore_index=True)
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

    print(f"\nSaved predictions: {output_path}")
    print(f"Saved metrics:     {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")

    if len(val_pred) > 0:
        s_ll = binary_log_loss(val_pred["score_label"].tolist(), val_pred["score_prob"].tolist())
        c_ll = binary_log_loss(val_pred["concede_label"].tolist(), val_pred["concede_prob"].tolist())
        print(f"Val score_log_loss={s_ll:.4f} | concede_log_loss={c_ll:.4f}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(SEED)
    use_cache = True  # 始终缓存 parquet feature df

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 获取文件列表并做 train/val 分割（文件级别，不加载 action dicts）
    # ------------------------------------------------------------------
    max_train = args.max_train_matches or MAX_TRAIN_MATCH_FILES
    max_test  = args.max_test_matches  or MAX_TEST_MATCH_FILES

    train_files_all = list_event_files(TRAIN_ROOT, max_match_files=max_train)
    train_files, val_files = _split_files_train_val(train_files_all, VAL_RATIO, SEED)
    test_files = list_event_files(TEST_ROOT, max_match_files=max_test)
    print(f"Files — train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")

    # ------------------------------------------------------------------
    # Eval-only fast path
    # ------------------------------------------------------------------
    if args.eval_only:
        if not os.path.exists(SCORE_MODEL_PATH) or not os.path.exists(CONCEDE_MODEL_PATH):
            raise FileNotFoundError(
                f"Models not found: {SCORE_MODEL_PATH}, {CONCEDE_MODEL_PATH}. Run training first."
            )
        print("Eval-only mode: loading saved models...")
        score_model = CatBoostClassifier()
        score_model.load_model(SCORE_MODEL_PATH)
        concede_model = CatBoostClassifier()
        concede_model.load_model(CONCEDE_MODEL_PATH)

        def _pred_from_files(name, files):
            if not files:
                return pd.DataFrame(), pd.DataFrame()
            df = load_or_build_feature_df(name, files, lookahead_k=10, use_cache=use_cache)
            s = predict_with_model(score_model,   df, "score_label",   "score_prob")
            c = predict_with_model(concede_model, df, "concede_label", "concede_prob")
            del df; gc.collect()
            return s, c

        train_dfs = _pred_from_files("train", train_files) if "train" in valid_splits else (pd.DataFrame(), pd.DataFrame())
        val_dfs   = _pred_from_files("val",   val_files)
        test_dfs  = _pred_from_files("test",  test_files)

        _export(args, valid_splits, train_dfs, val_dfs, test_dfs)
        return

    # ------------------------------------------------------------------
    # Training path — 逐个 split 加载，用完即释放，峰值内存 ~4GB
    # ------------------------------------------------------------------
    print("\nLoading train split...")
    train_df = load_or_build_feature_df("train", train_files, lookahead_k=10, use_cache=use_cache)
    print(f"  train rows: {len(train_df):,}")

    # 逐个 split 转换并立即释放 list，避免 list + DataFrame 同时占内存
    print("\nLoading val split...")
    val_df = load_or_build_feature_df("val", val_files, lookahead_k=10, use_cache=use_cache)
    print(f"  val rows:   {len(val_df):,}")

    val_score_pool   = make_pool(val_df, "score_label")
    val_concede_pool = make_pool(val_df, "concede_label")

    # 训练 score 模型：建 pool → 训练 → 立即释放 pool
    train_score_pool = make_pool(train_df, "score_label")
    score_model = train_catboost(train_score_pool, val_score_pool, "score", args)
    del train_score_pool; gc.collect()

    # 训练 concede 模型：建 pool → 训练 → 立即释放 pool
    train_concede_pool = make_pool(train_df, "concede_label")
    concede_model = train_catboost(train_concede_pool, val_concede_pool, "concede", args)
    del train_concede_pool, val_score_pool, val_concede_pool; gc.collect()

    # Save models
    score_model.save_model(SCORE_MODEL_PATH)
    concede_model.save_model(CONCEDE_MODEL_PATH)
    print(f"\nSaved: {SCORE_MODEL_PATH}")
    print(f"Saved: {CONCEDE_MODEL_PATH}")

    # Predict val / test（用完即释放，节省内存）
    def _pred(df):
        s = predict_with_model(score_model,   df, "score_label",   "score_prob")
        c = predict_with_model(concede_model, df, "concede_label", "concede_prob")
        return s, c

    train_dfs = _pred(train_df) if "train" in valid_splits else (pd.DataFrame(), pd.DataFrame())
    del train_df; gc.collect()

    val_dfs = _pred(val_df)
    del val_df; gc.collect()

    print("\nLoading test split...")
    test_df = load_or_build_feature_df("test", test_files, lookahead_k=10, use_cache=use_cache)
    print(f"  test rows:  {len(test_df):,}")
    test_dfs = _pred(test_df)
    del test_df; gc.collect()

    _export(args, valid_splits, train_dfs, val_dfs, test_dfs)


if __name__ == "__main__":
    main()
