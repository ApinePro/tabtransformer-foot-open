"""
train_vaep_xgboost.py

用 XGBoost 做 VAEP score/concede 双任务预测（各训一个独立分类器）。

与其他 VAEP 脚本的关系：
  - train_vaep_baseline.py       : 自训 embedding MLP
  - train_vaep_tabtransformer.py : 冻结预训练 encoder + MLP
  - train_vaep_catboost.py       : CatBoost
  - train_vaep_xgboost.py        : XGBoost（本脚本）

特点：
  - 直接复用 train_vaep_catboost.py 的 parquet 特征缓存（无需重新处理原始数据）
  - 类别列转为 pandas Categorical dtype + enable_categorical=True（XGBoost ≥ 1.6）
  - GPU 加速：device='cuda'

用法：
  python train_vaep_xgboost.py
  python train_vaep_xgboost.py --gpu
  python train_vaep_xgboost.py --eval-only
"""

import argparse
import gc
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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
from train_vaep_catboost import (
    _split_files_train_val,
    load_or_build_feature_df,
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
SCORE_MODEL_PATH  = os.path.join(CHECKPOINT_DIR, "vaep_xgboost_score.ubj")
CONCEDE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vaep_xgboost_concede.ubj")
PREDICTIONS_OUTPUT = "vaep_xgboost_predictions"
METRICS_OUTPUT = "vaep_xgboost_metrics.csv"

ITERATIONS = 1000
MAX_DEPTH  = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
EARLY_STOPPING_ROUNDS = 50
MIN_CHILD_WEIGHT = 5


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAEP with XGBoost")
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--depth", type=int, default=MAX_DEPTH)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--early-stopping", type=int, default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--prediction-splits", default="val,test")
    p.add_argument("--prediction-format", default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches",  type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 特征准备：类别列 → pd.Categorical（XGBoost enable_categorical 要求）
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """把 cat 列转为 Categorical dtype，XGBoost 才能识别。"""
    df = df.copy()
    for col in DEFAULT_CAT_COLS:
        df[col] = df[col].astype("category")
    for col in DEFAULT_CONT_COLS:
        df[col] = df[col].astype(np.float32)
    return df


def make_dmatrix(df: pd.DataFrame, label_col: str) -> xgb.DMatrix:
    X = prepare_features(df[DEFAULT_CAT_COLS + DEFAULT_CONT_COLS])
    y = df[label_col].values.astype(np.float32)
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    pos = labels.sum()
    neg = len(labels) - pos
    return float(neg / pos) if pos > 0 else 1.0


# ---------------------------------------------------------------------------
def train_xgboost(
    train_dm: xgb.DMatrix,
    val_dm: xgb.DMatrix,
    scale_pos_weight: float,
    label: str,
    args: argparse.Namespace,
) -> xgb.Booster:
    device = "cuda" if args.gpu else "cpu"
    params = {
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "max_depth":         args.depth,
        "eta":               args.lr,
        "subsample":         SUBSAMPLE,
        "colsample_bytree":  COLSAMPLE_BYTREE,
        "min_child_weight":  MIN_CHILD_WEIGHT,
        "scale_pos_weight":  scale_pos_weight,
        "tree_method":       "hist",
        "device":            device,
        "seed":              SEED,
        "verbosity":         1,
    }
    callbacks = [xgb.callback.EarlyStopping(
        rounds=args.early_stopping,
        metric_name="auc",
        data_name="val",
        save_best=True,
        maximize=True,
    )]
    print(f"\nTraining XGBoost for {label} (scale_pos_weight={scale_pos_weight:.1f}) ...")
    booster = xgb.train(
        params,
        train_dm,
        num_boost_round=args.iterations,
        evals=[(val_dm, "val")],
        verbose_eval=100,
        callbacks=callbacks,
    )
    return booster


def predict_with_booster(
    booster: xgb.Booster,
    df: pd.DataFrame,
    label_col: str,
    prob_col: str,
) -> pd.DataFrame:
    X = prepare_features(df[DEFAULT_CAT_COLS + DEFAULT_CONT_COLS])
    dm = xgb.DMatrix(X, enable_categorical=True)
    probs = booster.predict(dm)
    return pd.DataFrame({
        "match_id":     df["match_id"].values,
        "source_name":  df["source_name"].values,
        "action_index": df["action_index"].values,
        label_col:      df[label_col].values,
        prob_col:       probs,
    })


# ---------------------------------------------------------------------------
def _export(args, valid_splits, train_dfs, val_dfs, test_dfs):
    def merge_preds(score_df, concede_df):
        if len(score_df) == 0:
            return pd.DataFrame()
        return score_df.merge(
            concede_df[["match_id", "action_index", "concede_label", "concede_prob"]],
            on=["match_id", "action_index"],
            how="inner",
        )

    val_pred  = merge_preds(*val_dfs)
    test_pred = merge_preds(*test_dfs)
    train_pred = merge_preds(*train_dfs) if train_dfs else pd.DataFrame()

    if len(val_pred) > 0:
        score_thr, _   = find_best_threshold(val_pred["score_label"].astype(int).tolist(),
                                              val_pred["score_prob"].astype(float).tolist())
        concede_thr, _ = find_best_threshold(val_pred["concede_label"].astype(int).tolist(),
                                              val_pred["concede_prob"].astype(float).tolist())
    else:
        score_thr, concede_thr = 0.5, 0.5

    for df, name in [(train_pred, "train"), (val_pred, "val"), (test_pred, "test")]:
        if len(df) > 0:
            df["split"] = name

    pred_map = {"train": train_pred, "val": val_pred, "test": test_pred}
    parts = [pred_map[s] for s in valid_splits if len(pred_map.get(s, pd.DataFrame())) > 0]
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

    print(f"\nSaved predictions: {output_path}")
    print(f"Saved metrics:     {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")

    if len(val_pred) > 0:
        s_ll = binary_log_loss(val_pred["score_label"].tolist(),   val_pred["score_prob"].tolist())
        c_ll = binary_log_loss(val_pred["concede_label"].tolist(), val_pred["concede_prob"].tolist())
        print(f"Val score_log_loss={s_ll:.4f} | concede_log_loss={c_ll:.4f}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(SEED)

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
            raise FileNotFoundError(f"Models not found. Run training first.")
        score_booster = xgb.Booster()
        score_booster.load_model(SCORE_MODEL_PATH)
        concede_booster = xgb.Booster()
        concede_booster.load_model(CONCEDE_MODEL_PATH)

        def _pred_from_files(name, files):
            if not files:
                return pd.DataFrame(), pd.DataFrame()
            df = load_or_build_feature_df(name, files, lookahead_k=10, use_cache=True)
            s = predict_with_booster(score_booster,   df, "score_label",   "score_prob")
            c = predict_with_booster(concede_booster, df, "concede_label", "concede_prob")
            del df; gc.collect()
            return s, c

        train_dfs = _pred_from_files("train", train_files) if "train" in valid_splits else None
        val_dfs   = _pred_from_files("val",   val_files)
        test_dfs  = _pred_from_files("test",  test_files)
        _export(args, valid_splits, train_dfs, val_dfs, test_dfs)
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    print("\nLoading train split...")
    train_df = load_or_build_feature_df("train", train_files, lookahead_k=10, use_cache=True)
    print(f"  train rows: {len(train_df):,}")

    print("\nLoading val split...")
    val_df = load_or_build_feature_df("val", val_files, lookahead_k=10, use_cache=True)
    print(f"  val rows:   {len(val_df):,}")

    # Build DMatrices（val 共用）
    val_score_dm   = make_dmatrix(val_df, "score_label")
    val_concede_dm = make_dmatrix(val_df, "concede_label")

    # Train score
    spw_score = compute_scale_pos_weight(train_df["score_label"].values)
    train_score_dm = make_dmatrix(train_df, "score_label")
    score_booster = train_xgboost(train_score_dm, val_score_dm, spw_score, "score", args)
    del train_score_dm; gc.collect()

    # Train concede
    spw_concede = compute_scale_pos_weight(train_df["concede_label"].values)
    train_concede_dm = make_dmatrix(train_df, "concede_label")
    concede_booster = train_xgboost(train_concede_dm, val_concede_dm, spw_concede, "concede", args)
    del train_concede_dm, val_score_dm, val_concede_dm; gc.collect()

    # Save models
    score_booster.save_model(SCORE_MODEL_PATH)
    concede_booster.save_model(CONCEDE_MODEL_PATH)
    print(f"\nSaved: {SCORE_MODEL_PATH}")
    print(f"Saved: {CONCEDE_MODEL_PATH}")

    # Predict
    def _pred(df):
        s = predict_with_booster(score_booster,   df, "score_label",   "score_prob")
        c = predict_with_booster(concede_booster, df, "concede_label", "concede_prob")
        return s, c

    train_dfs = _pred(train_df) if "train" in valid_splits else (pd.DataFrame(), pd.DataFrame())
    del train_df; gc.collect()

    val_dfs = _pred(val_df)
    del val_df; gc.collect()

    print("\nLoading test split...")
    test_df = load_or_build_feature_df("test", test_files, lookahead_k=10, use_cache=True)
    print(f"  test rows:  {len(test_df):,}")
    test_dfs = _pred(test_df)
    del test_df; gc.collect()

    _export(args, valid_splits, train_dfs, val_dfs, test_dfs)


if __name__ == "__main__":
    main()
