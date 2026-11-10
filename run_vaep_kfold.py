"""
run_vaep_kfold.py — VAEP 任务（score+concede）的 match-level K-fold CV runner。

模型（--models 逗号分隔，按便宜→贵排序执行）:
  tt_catboost_pca : 冻结 encoder 的 PCA-128 embedding（复用缓存, 口径C）+ CatBoost
  catboost_raw    : StatsBomb 44 维原始特征 + CatBoost
  xgboost_raw     : StatsBomb 44 维原始特征 + XGBoost
  mlp_baseline    : 随机 emb 端到端 VAEPMultitaskMLP（8ep, 选 checkpoint=val 平均 LogLoss）
  tt_mlp          : 冻结 encoder 911 维 embedding（复用 memmap）+ FrozenEncoderVAEP 同构 MLP 头（10ep）

协议:
  - K=5 match-level fold，FOLD_SEED=42（vaep_kfold_fold_assignments.csv 落盘复现）
  - 每折训练 seed = 1000 + fold；test 固定为三赛事 526K 动作
  - 超参全部沿用各原始脚本默认值；神经模型 checkpoint 按原协议
    （val 上 score/concede LogLoss 均值最低）选择
  - 缓存复用：cb_*.parquet 特征表、PCA-128/911 维 embedding memmap（冻结 encoder
    输出与 fold 划分无关，等价固定特征提取器）

抗中断: 与 run_xg_kfold.py 同款（单元级落盘+跳过、原子写、异常隔离）。

用法:
  python run_vaep_kfold.py --models tt_catboost_pca,catboost_raw,xgboost_raw,mlp_baseline,tt_mlp --gpu
"""

import argparse
import gc
import glob
import hashlib
import math
import os
import random
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import DEFAULT_CAT_COLS, DEFAULT_CONT_COLS, set_seed
from train_vaep_baseline import (
    CACHE_DIR,
    SEED,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    WEIGHT_DECAY,
    VAEPMultitaskMLP,
    binary_log_loss,
    binary_roc_auc,
    binary_f1_score,
    brier_score,
    compute_pos_weight,
    find_best_threshold,
    list_event_files,
)
import train_vaep_baseline as vb
from train_vaep_catboost import _split_files_train_val, load_or_build_feature_df
from train_vaep_spadl_baseline import (
    CONTEXT_SIZE as SPADL_CTX,
    _get_feature_cols as spadl_feature_cols,
    load_or_build_spadl_df,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
K_FOLDS         = 5
FOLD_SEED       = 42
TRAIN_SEED_BASE = 1000

FOLD_ASSIGN_CSV  = "vaep_kfold_fold_assignments.csv"
RESULTS_DIR      = "vaep_kfold_results"
COMBINED_METRICS = "vaep_kfold_metrics.csv"
SUMMARY_CSV      = "vaep_kfold_summary.csv"
EMBED_DIR        = os.path.join(CACHE_DIR, "tt_embeddings")

ALL_MODELS  = ["tt_catboost_pca", "catboost_raw", "xgboost_raw", "mlp_baseline", "tt_mlp",
               "spadl_catboost", "spadl_xgboost"]
METRIC_COLS = ["roc_auc", "log_loss", "brier", "f1"]
TASKS       = ["score", "concede"]

# 树模型超参（train_vaep_catboost / train_vaep_xgboost 默认值）
CB_ITERATIONS, CB_DEPTH, CB_LR, CB_L2, CB_EARLY = 1000, 6, 0.1, 3.0, 50
XGB_SUBSAMPLE, XGB_COLSAMPLE, XGB_MIN_CHILD = 0.9, 0.8, 5
# 神经模型超参（各自脚本默认值）
MLP_EPOCHS, MLP_BATCH, MLP_LR = 8, 256, 2e-4
TTMLP_EPOCHS, TTMLP_BATCH, TTMLP_LR, TTMLP_HIDDEN = 10, 512, 1e-3, 256
BLOCK_ROWS = 65536  # tt_mlp memmap 顺序块大小


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Match-level K-fold CV for VAEP models")
    p.add_argument("--models", default="tt_catboost_pca,catboost_raw,xgboost_raw,mlp_baseline,tt_mlp")
    p.add_argument("--k", type=int, default=K_FOLDS)
    p.add_argument("--folds", default=None)
    p.add_argument("--gpu", action="store_true", help="GPU for CatBoost/XGBoost")
    p.add_argument("--iterations", type=int, default=CB_ITERATIONS)
    p.add_argument("--nn-epochs", type=int, default=None, help="Override NN epochs (smoke only)")
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    compression = "gzip" if path.endswith(".gz") else None
    df.to_csv(tmp, index=False, encoding="utf-8-sig", compression=compression)
    os.replace(tmp, path)


def cell_paths(results_dir: str, model: str, fold: int) -> Tuple[str, str, str]:
    base = os.path.join(results_dir, f"vaep_kfold_{model}_fold{fold}")
    return base + "_metrics.csv", base + "_predictions.csv.gz", base + "_ERROR.txt"


# ---------------------------------------------------------------------------
# Data loading (all cache-backed)
# ---------------------------------------------------------------------------
def load_pool_and_test_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """全池 44 维特征表 = 原 catboost run 的 train+val parquet 缓存拼接。"""
    train_files_all = list_event_files(TRAIN_ROOT, max_match_files=None)
    train_files, val_files = _split_files_train_val(train_files_all, VAL_RATIO, SEED)
    test_files = list_event_files(TEST_ROOT, max_match_files=None)

    df_train = load_or_build_feature_df("train", train_files, 10, use_cache=True)
    df_val   = load_or_build_feature_df("val",   val_files,   10, use_cache=True)
    df_test  = load_or_build_feature_df("test",  test_files,  10, use_cache=True)
    pool = pd.concat([df_train, df_val], ignore_index=True)
    del df_train, df_val
    gc.collect()
    print(f"pool actions = {len(pool):,} | test actions = {len(df_test):,} | "
          f"pool matches = {pool['match_id'].nunique()}")
    return pool, df_test


def load_spadl_pool_and_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SPADL 27维特征表 = 原 SPADL run 的 train+val parquet 缓存拼接。"""
    train_files_all = list_event_files(TRAIN_ROOT, max_match_files=None)
    train_files, val_files = _split_files_train_val(train_files_all, VAL_RATIO, SEED)
    test_files = list_event_files(TEST_ROOT, max_match_files=None)
    df_train = load_or_build_spadl_df("train", train_files, 10, SPADL_CTX, use_cache=True)
    df_val   = load_or_build_spadl_df("val",   val_files,   10, SPADL_CTX, use_cache=True)
    df_test  = load_or_build_spadl_df("test",  test_files,  10, SPADL_CTX, use_cache=True)
    pool = pd.concat([df_train, df_val], ignore_index=True)
    del df_train, df_val
    gc.collect()
    print(f"SPADL pool actions = {len(pool):,} | test = {len(df_test):,} | "
          f"matches = {pool['match_id'].nunique()}")
    return pool, df_test


def _find_one(pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"Cache not found: {pattern}")
    return sorted(hits)[0]


def load_embedding_bundle(tag: str) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """加载 emb_{train|val|test} memmap（911 或 pca128）+ labels。返回 (emb, score, concede, match_ids)。"""
    bin_path = _find_one(os.path.join(EMBED_DIR, f"emb_{tag}.bin"))
    shape = tuple(np.load(bin_path.replace(".bin", "_shape.npy")).tolist())
    emb = np.memmap(bin_path, dtype="float32", mode="r", shape=shape)
    labels = np.load(bin_path.replace(".bin", "_labels.npz"), allow_pickle=True)
    return emb, labels["score_labels"], labels["concede_labels"], labels["match_ids"].astype(np.int64)


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------
def get_fold_assignments(match_ids: np.ndarray, k: int) -> Dict[int, int]:
    if os.path.exists(FOLD_ASSIGN_CSV):
        df = pd.read_csv(FOLD_ASSIGN_CSV)
        print(f"Loaded fold assignments: {FOLD_ASSIGN_CSV} ({df['match_id'].nunique()} matches)")
        return dict(zip(df["match_id"].astype(int), df["fold"].astype(int)))
    ids = sorted(set(int(m) for m in match_ids))
    rng = random.Random(FOLD_SEED)
    rng.shuffle(ids)
    chunks = np.array_split(np.asarray(ids), k)
    assign = {int(m): f for f, chunk in enumerate(chunks) for m in chunk}
    atomic_write_csv(pd.DataFrame(sorted(assign.items()), columns=["match_id", "fold"]), FOLD_ASSIGN_CSV)
    print(f"Created fold assignments (FOLD_SEED={FOLD_SEED}, k={k}) → {FOLD_ASSIGN_CSV}")
    return assign


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def task_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y = labels.astype(int).tolist()
    p = probs.astype(float).tolist()
    preds = (np.asarray(p) >= threshold).astype(int).tolist()
    return {
        "roc_auc": binary_roc_auc(y, p),
        "log_loss": binary_log_loss(y, p),
        "brier": brier_score(y, p),
        "f1": binary_f1_score(y, preds),
        "threshold": threshold,
        "positive_rate": float(np.mean(labels)),
    }


def write_cell(results_dir, model, fold, seed, task_split_results, pred_df) -> None:
    metrics_path, pred_path, error_path = cell_paths(results_dir, model, fold)
    rows = []
    for (task, split), m in task_split_results.items():
        rows.append({"model": model, "fold": fold, "seed": seed, "task": task, "split": split,
                     "n": m.pop("n"), **m})
        print(f"[{model} f{fold}] {task}/{split}: auc={m['roc_auc']:.4f} ll={m['log_loss']:.4f} "
              f"brier={m['brier']:.4f} f1={m['f1']:.4f}")
    if pred_df is not None:
        atomic_write_csv(pred_df, pred_path)
    atomic_write_csv(pd.DataFrame(rows), metrics_path)
    if os.path.exists(error_path):
        os.remove(error_path)
    print(f"[DONE] {model} fold{fold} → {metrics_path}")


def evaluate_both_tasks(val_out: Dict[str, np.ndarray], test_out: Dict[str, np.ndarray]) -> Tuple[dict, pd.DataFrame]:
    """val_out/test_out: dict with score_label/score_prob/concede_label/concede_prob (+match_id, action_index for test)."""
    results = {}
    for task in TASKS:
        thr, _ = find_best_threshold(val_out[f"{task}_label"].astype(int).tolist(),
                                     val_out[f"{task}_prob"].astype(float).tolist())
        m_val = task_metrics(val_out[f"{task}_label"], val_out[f"{task}_prob"], thr)
        m_val["n"] = len(val_out[f"{task}_label"])
        m_test = task_metrics(test_out[f"{task}_label"], test_out[f"{task}_prob"], thr)
        m_test["n"] = len(test_out[f"{task}_label"])
        results[(task, "val")] = m_val
        results[(task, "test")] = m_test
    pred_df = pd.DataFrame({
        "match_id": test_out["match_id"], "action_index": test_out["action_index"],
        "score_label": test_out["score_label"].astype(int), "score_prob": test_out["score_prob"],
        "concede_label": test_out["concede_label"].astype(int), "concede_prob": test_out["concede_prob"],
    })
    return results, pred_df


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------
def fit_catboost(X_train, y_train, X_val, y_val, cat_idx, gpu: bool, iterations: int, seed: int):
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(
        iterations=iterations, depth=CB_DEPTH, learning_rate=CB_LR, l2_leaf_reg=CB_L2,
        loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        task_type="GPU" if gpu else "CPU", early_stopping_rounds=CB_EARLY,
        verbose=200, auto_class_weights="Balanced",
    )
    model.fit(Pool(X_train, y_train, cat_features=cat_idx),
              eval_set=Pool(X_val, y_val, cat_features=cat_idx), use_best_model=True)
    return model


def run_catboost_raw(fold, tr_df, va_df, test_df, gpu, iterations, seed,
                     cat_cols=DEFAULT_CAT_COLS, cont_cols=DEFAULT_CONT_COLS):
    feat = list(cat_cols) + list(cont_cols)
    cat_idx = list(range(len(cat_cols)))
    out_val, out_test = {}, {}
    for task in TASKS:
        lbl = f"{task}_label"
        model = fit_catboost(tr_df[feat], tr_df[lbl].values, va_df[feat], va_df[lbl].values,
                             cat_idx, gpu, iterations, seed)
        out_val[f"{task}_prob"] = model.predict_proba(va_df[feat])[:, 1]
        out_val[f"{task}_label"] = va_df[lbl].values
        out_test[f"{task}_prob"] = model.predict_proba(test_df[feat])[:, 1]
        out_test[f"{task}_label"] = test_df[lbl].values
        del model
        gc.collect()
    out_test["match_id"] = test_df["match_id"].values
    out_test["action_index"] = test_df["action_index"].values
    return out_val, out_test


def run_xgboost_raw(fold, tr_df, va_df, test_df, gpu, iterations, seed,
                    cat_cols=DEFAULT_CAT_COLS, cont_cols=DEFAULT_CONT_COLS):
    import xgboost as xgb
    feat = list(cat_cols) + list(cont_cols)

    def prep(df):
        X = df[feat].copy()
        for c in cat_cols:
            X[c] = X[c].astype("category")
        for c in cont_cols:
            X[c] = X[c].astype(np.float32)
        return X

    Xtr, Xva, Xte = prep(tr_df), prep(va_df), prep(test_df)
    out_val, out_test = {}, {}
    for task in TASKS:
        lbl = f"{task}_label"
        y_tr = tr_df[lbl].values.astype(np.float32)
        spw = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
        params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": CB_DEPTH,
                  "eta": CB_LR, "subsample": XGB_SUBSAMPLE, "colsample_bytree": XGB_COLSAMPLE,
                  "min_child_weight": XGB_MIN_CHILD, "scale_pos_weight": spw,
                  "tree_method": "hist", "device": "cuda" if gpu else "cpu", "seed": seed}
        dtr = xgb.DMatrix(Xtr, label=y_tr, enable_categorical=True)
        dva = xgb.DMatrix(Xva, label=va_df[lbl].values.astype(np.float32), enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=iterations, evals=[(dva, "val")],
                            verbose_eval=200,
                            callbacks=[xgb.callback.EarlyStopping(rounds=CB_EARLY, metric_name="auc",
                                                                  data_name="val", save_best=True, maximize=True)])
        out_val[f"{task}_prob"] = booster.predict(dva)
        out_val[f"{task}_label"] = va_df[lbl].values
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        out_test[f"{task}_prob"] = booster.predict(dte)
        out_test[f"{task}_label"] = test_df[lbl].values
        del booster, dtr, dva, dte
        gc.collect()
    out_test["match_id"] = test_df["match_id"].values
    out_test["action_index"] = test_df["action_index"].values
    return out_val, out_test


def run_tt_catboost_pca(fold, pool_emb, pool_score, pool_concede, pool_mids, tr_mask, va_mask,
                        test_emb, test_score, test_concede, test_mids, test_aidx,
                        gpu, iterations, seed):
    out_val, out_test = {}, {}
    Xtr, Xva = pool_emb[tr_mask], pool_emb[va_mask]
    for task, y_pool, y_test in [("score", pool_score, test_score), ("concede", pool_concede, test_concede)]:
        model = fit_catboost(Xtr, y_pool[tr_mask], Xva, y_pool[va_mask], [], gpu, iterations, seed)
        out_val[f"{task}_prob"] = model.predict_proba(Xva)[:, 1]
        out_val[f"{task}_label"] = y_pool[va_mask]
        out_test[f"{task}_prob"] = model.predict_proba(np.asarray(test_emb))[:, 1]
        out_test[f"{task}_label"] = y_test
        del model
        gc.collect()
    out_test["match_id"] = test_mids
    out_test["action_index"] = test_aidx
    return out_val, out_test


# ---------------------------------------------------------------------------
# Neural: mlp_baseline (vectorized array path, original hyperparams)
# ---------------------------------------------------------------------------
def encode_cats(df_col: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    return df_col.map(mapping).fillna(0).to_numpy(dtype=np.int64)


def nn_predict(model, cat_t, cont_t, batch: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    s_probs, c_probs = [], []
    with torch.no_grad():
        for i in range(0, len(cat_t), batch):
            xc = cat_t[i:i + batch].to(DEVICE, non_blocking=True)
            xn = cont_t[i:i + batch].to(DEVICE, non_blocking=True)
            s_logit, c_logit = model(xc, xn)
            s_probs.append(torch.sigmoid(s_logit).cpu().numpy())
            c_probs.append(torch.sigmoid(c_logit).cpu().numpy())
    return np.concatenate(s_probs), np.concatenate(c_probs)


def run_mlp_baseline(fold, tr_df, va_df, test_df, nn_epochs, seed):
    epochs = nn_epochs or MLP_EPOCHS
    cat_maps = {}
    for col in DEFAULT_CAT_COLS:
        uniques = sorted(tr_df[col].astype(str).unique())
        cat_maps[col] = {v: i + 1 for i, v in enumerate(uniques)}
    cat_dims = [len(cat_maps[c]) + 1 for c in DEFAULT_CAT_COLS]
    stats = []
    for col in DEFAULT_CONT_COLS:
        arr = tr_df[col].to_numpy(dtype=np.float32)
        std = float(arr.std()) if len(arr) > 1 else 1.0
        stats.append((float(arr.mean()), std if std > 1e-6 else 1.0))
    cont_mean_std = np.asarray(stats, dtype=np.float32)

    def to_tensors(df):
        cat = np.stack([encode_cats(df[c].astype(str), cat_maps[c]) for c in DEFAULT_CAT_COLS], axis=1)
        cont = df[DEFAULT_CONT_COLS].to_numpy(dtype=np.float32)
        return torch.from_numpy(cat), torch.from_numpy(cont)

    tr_cat, tr_cont = to_tensors(tr_df)
    va_cat, va_cont = to_tensors(va_df)
    te_cat, te_cont = to_tensors(test_df)
    tr_s = torch.from_numpy(tr_df["score_label"].to_numpy(dtype=np.float32))
    tr_c = torch.from_numpy(tr_df["concede_label"].to_numpy(dtype=np.float32))

    model = VAEPMultitaskMLP(cat_dims, len(DEFAULT_CONT_COLS), cont_mean_std).to(DEVICE)
    s_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        compute_pos_weight(tr_df["score_label"].astype(int).tolist()), device=DEVICE))
    c_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        compute_pos_weight(tr_df["concede_label"].astype(int).tolist()), device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)

    best_obj, best_state = float("inf"), None
    n = len(tr_cat)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, MLP_BATCH):
            idx = perm[i:i + MLP_BATCH]
            xc = tr_cat[idx].to(DEVICE, non_blocking=True)
            xn = tr_cont[idx].to(DEVICE, non_blocking=True)
            ys = tr_s[idx].to(DEVICE, non_blocking=True)
            yc = tr_c[idx].to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            s_logit, c_logit = model(xc, xn)
            loss = s_crit(s_logit, ys) + c_crit(c_logit, yc)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        s_prob, c_prob = nn_predict(model, va_cat, va_cont, 8192)
        s_ll = binary_log_loss(va_df["score_label"].astype(int).tolist(), s_prob.tolist())
        c_ll = binary_log_loss(va_df["concede_label"].astype(int).tolist(), c_prob.tolist())
        obj = 0.5 * (s_ll + c_ll)
        print(f"[mlp_baseline f{fold}] Epoch {epoch:02d} | loss={total / max(n // MLP_BATCH, 1):.4f} | "
              f"val_score_ll={s_ll:.4f} | val_concede_ll={c_ll:.4f}")
        if obj < best_obj:
            best_obj = obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    vs, vc = nn_predict(model, va_cat, va_cont, 8192)
    ts, tc = nn_predict(model, te_cat, te_cont, 8192)
    out_val = {"score_prob": vs, "score_label": va_df["score_label"].values,
               "concede_prob": vc, "concede_label": va_df["concede_label"].values}
    out_test = {"score_prob": ts, "score_label": test_df["score_label"].values,
                "concede_prob": tc, "concede_label": test_df["concede_label"].values,
                "match_id": test_df["match_id"].values, "action_index": test_df["action_index"].values}
    del model, tr_cat, tr_cont, va_cat, va_cont, te_cat, te_cont
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out_val, out_test


# ---------------------------------------------------------------------------
# Neural: tt_mlp (precomputed 911-dim memmap + FrozenEncoderVAEP-style head)
# ---------------------------------------------------------------------------
class VAEPHead(nn.Module):
    """FrozenEncoderVAEP 的 mlp+双头部分，输入为预计算的 911 维 embedding。"""

    def __init__(self, embed_dim: int, hidden_dim: int = TTMLP_HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(0.1),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.concede_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(emb)
        return self.score_head(h).squeeze(-1), self.concede_head(h).squeeze(-1)


def head_predict(model: VAEPHead, emb_arr: np.ndarray, batch: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    s_probs, c_probs = [], []
    with torch.no_grad():
        for i in range(0, len(emb_arr), batch):
            x = torch.from_numpy(np.ascontiguousarray(emb_arr[i:i + batch])).to(DEVICE)
            s_logit, c_logit = model(x)
            s_probs.append(torch.sigmoid(s_logit).cpu().numpy())
            c_probs.append(torch.sigmoid(c_logit).cpu().numpy())
    return np.concatenate(s_probs), np.concatenate(c_probs)


def run_tt_mlp(fold, segments, tr_masks, va_masks, test_emb, test_score, test_concede,
               test_mids, test_aidx, nn_epochs, seed):
    """segments: list of (memmap, score, concede) 覆盖全池; masks 与 segment 对齐."""
    epochs = nn_epochs or TTMLP_EPOCHS
    embed_dim = segments[0][0].shape[1]

    # 物化 fold-val embedding 到 RAM（~4.6GB），train 走顺序块扫描
    va_emb = np.concatenate([np.asarray(seg[0][mask]) for (seg, mask) in zip(segments, va_masks)], axis=0)
    va_s = np.concatenate([seg[1][mask] for (seg, mask) in zip(segments, va_masks)])
    va_c = np.concatenate([seg[2][mask] for (seg, mask) in zip(segments, va_masks)])

    tr_s_all = np.concatenate([seg[1][mask] for (seg, mask) in zip(segments, tr_masks)])
    tr_c_all = np.concatenate([seg[2][mask] for (seg, mask) in zip(segments, tr_masks)])

    model = VAEPHead(embed_dim).to(DEVICE)
    s_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        compute_pos_weight(tr_s_all.astype(int).tolist()), device=DEVICE))
    c_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        compute_pos_weight(tr_c_all.astype(int).tolist()), device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TTMLP_LR, weight_decay=WEIGHT_DECAY)
    del tr_s_all, tr_c_all

    # 预生成 (segment_id, block_start) 列表用于块级打乱（顺序 I/O + 块内打乱）
    blocks = []
    for si, (seg, _) in enumerate(zip(segments, tr_masks)):
        for start in range(0, seg[0].shape[0], BLOCK_ROWS):
            blocks.append((si, start))

    best_obj, best_state = float("inf"), None
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(blocks)
        total, nb = 0.0, 0
        for si, start in blocks:
            emb_mm, s_arr, c_arr = segments[si]
            mask = tr_masks[si][start:start + BLOCK_ROWS]
            if not mask.any():
                continue
            block = np.asarray(emb_mm[start:start + BLOCK_ROWS])[mask]
            bs = s_arr[start:start + BLOCK_ROWS][mask].astype(np.float32)
            bc = c_arr[start:start + BLOCK_ROWS][mask].astype(np.float32)
            perm = rng.permutation(len(block))
            for i in range(0, len(block), TTMLP_BATCH):
                idx = perm[i:i + TTMLP_BATCH]
                x = torch.from_numpy(block[idx]).to(DEVICE)
                ys = torch.from_numpy(bs[idx]).to(DEVICE)
                yc = torch.from_numpy(bc[idx]).to(DEVICE)
                optimizer.zero_grad()
                s_logit, c_logit = model(x)
                loss = s_crit(s_logit, ys) + c_crit(c_logit, yc)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                nb += 1
        s_prob, c_prob = head_predict(model, va_emb)
        s_ll = binary_log_loss(va_s.astype(int).tolist(), s_prob.tolist())
        c_ll = binary_log_loss(va_c.astype(int).tolist(), c_prob.tolist())
        obj = 0.5 * (s_ll + c_ll)
        print(f"[tt_mlp f{fold}] Epoch {epoch:02d} | loss={total / max(nb, 1):.4f} | "
              f"val_score_ll={s_ll:.4f} | val_concede_ll={c_ll:.4f}")
        if obj < best_obj:
            best_obj = obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    vs, vc = head_predict(model, va_emb)
    ts, tc = head_predict(model, np.asarray(test_emb))
    out_val = {"score_prob": vs, "score_label": va_s, "concede_prob": vc, "concede_label": va_c}
    out_test = {"score_prob": ts, "score_label": test_score, "concede_prob": tc, "concede_label": test_concede,
                "match_id": test_mids, "action_index": test_aidx}
    del model, va_emb
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out_val, out_test


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def t_ppf_975(df_: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df_))
    except ImportError:
        return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(df_, 1.96)


def aggregate(results_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "vaep_kfold_*_fold*_metrics.csv")))
    if not files:
        print("No cell metrics found.")
        return
    combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    atomic_write_csv(combined, COMBINED_METRICS)

    test_df = combined[combined["split"] == "test"]
    rows = []
    print("\n================ VAEP K-FOLD SUMMARY (test set) ================")
    for (model, task), g in test_df.groupby(["model", "task"]):
        n = len(g)
        tcrit = t_ppf_975(n - 1) if n > 1 else float("nan")
        row = {"kind": "model", "model": model, "task": task, "n_folds": n}
        line = f"{model} [{task}] (n={n}): "
        for metric in METRIC_COLS:
            vals = g[metric].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n > 1 else 0.0
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd, f"{metric}_se": se,
                        f"{metric}_ci95_lo": mean - tcrit * se if n > 1 else float("nan"),
                        f"{metric}_ci95_hi": mean + tcrit * se if n > 1 else float("nan")})
            line += f"{metric}={mean:.4f}±{se:.4f} "
        rows.append(row)
        print(line)

    try:
        from scipy import stats as sps
    except ImportError:
        sps = None
    print("\n---------------- Paired comparisons (test) ----------------")
    for task in TASKS:
        tdf = test_df[test_df["task"] == task]
        models_done = sorted(tdf["model"].unique())
        for i in range(len(models_done)):
            for j in range(i + 1, len(models_done)):
                a, b = models_done[i], models_done[j]
                ga = tdf[tdf["model"] == a].set_index("fold").sort_index()
                gb = tdf[tdf["model"] == b].set_index("fold").sort_index()
                common = ga.index.intersection(gb.index)
                if len(common) < 2:
                    continue
                tcrit = t_ppf_975(len(common) - 1)
                row = {"kind": "paired_diff", "model": f"{b} - {a}", "task": task, "n_folds": len(common)}
                line = f"[{task}] {b} - {a} (n={len(common)}): "
                for metric in METRIC_COLS:
                    diffs = (gb.loc[common, metric].to_numpy(dtype=float)
                             - ga.loc[common, metric].to_numpy(dtype=float))
                    mean = float(np.mean(diffs))
                    sd = float(np.std(diffs, ddof=1))
                    se = sd / math.sqrt(len(common))
                    pval = float(sps.ttest_rel(diffs, np.zeros_like(diffs)).pvalue) if sps else float("nan")
                    row.update({f"{metric}_mean": mean, f"{metric}_se": se,
                                f"{metric}_ci95_lo": mean - tcrit * se, f"{metric}_ci95_hi": mean + tcrit * se,
                                f"{metric}_p_paired_t": pval})
                    line += f"Δ{metric}={mean:+.4f} (p={pval:.3f}) "
                rows.append(row)
                print(line)

    atomic_write_csv(pd.DataFrame(rows), SUMMARY_CSV)
    print(f"\nSaved: {COMBINED_METRICS}, {SUMMARY_CSV}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.results_dir)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in models if m not in ALL_MODELS]
    if invalid:
        raise SystemExit(f"Unknown models: {invalid}. Choose from {ALL_MODELS}")
    folds = list(range(args.k)) if args.folds is None else [int(f) for f in args.folds.split(",")]

    print(f"DEVICE = {DEVICE}")
    print(f"MODELS = {models} | K = {args.k} | FOLDS = {folds} | GPU-trees = {args.gpu}")
    print(f"FOLD_SEED = {FOLD_SEED} | TRAIN_SEED = {TRAIN_SEED_BASE}+fold")

    need_df = any(m in models for m in ("catboost_raw", "xgboost_raw", "mlp_baseline"))
    need_spadl = any(m in models for m in ("spadl_catboost", "spadl_xgboost"))
    need_pca = "tt_catboost_pca" in models
    need_911 = "tt_mlp" in models

    pool_df = test_df = None
    if need_df:
        pool_df, test_df = load_pool_and_test_df()

    spadl_pool = spadl_test = None
    if need_spadl:
        spadl_pool, spadl_test = load_spadl_pool_and_test()

    pca = None
    if need_pca:
        tr = load_embedding_bundle("train_*_pca128")
        va = load_embedding_bundle("val_*_pca128")
        te = load_embedding_bundle("test_*_pca128")
        pool_emb = np.concatenate([np.asarray(tr[0]), np.asarray(va[0])], axis=0)
        pca = {
            "emb": pool_emb,
            "score": np.concatenate([tr[1], va[1]]),
            "concede": np.concatenate([tr[2], va[2]]),
            "mids": np.concatenate([tr[3], va[3]]),
            "test": te,
        }
        del tr, va
        gc.collect()
        print(f"PCA-128 pool: {pca['emb'].shape} | test: {pca['test'][0].shape}")

    seg911 = None
    if need_911:
        tr = load_embedding_bundle("train_7c9bae65b5a9")
        va = load_embedding_bundle("val_7252528aa267")
        te = load_embedding_bundle("test_d17dd2b7996a")
        seg911 = {"segments": [(tr[0], tr[1], tr[2]), (va[0], va[1], va[2])],
                  "seg_mids": [tr[3], va[3]], "test": te}
        print(f"911-dim segments: {tr[0].shape} + {va[0].shape} | test {te[0].shape}")

    # fold assignment over union of pool match_ids
    if pool_df is not None:
        all_mids = pool_df["match_id"].unique()
    elif spadl_pool is not None:
        all_mids = spadl_pool["match_id"].unique()
    elif pca is not None:
        all_mids = np.unique(pca["mids"])
    else:
        all_mids = np.unique(np.concatenate(seg911["seg_mids"]))
    assign = get_fold_assignments(all_mids, args.k)

    # test action_index fallbacks for embedding-only paths
    def emb_test_meta(te_bundle):
        n = te_bundle[0].shape[0]
        return te_bundle[3], np.arange(n, dtype=np.int64)

    for fold in folds:
        for model_name in models:
            metrics_path, _, error_path = cell_paths(args.results_dir, model_name, fold)
            if os.path.exists(metrics_path):
                print(f"[SKIP] {model_name} fold{fold} already done")
                continue
            seed = TRAIN_SEED_BASE + fold
            set_seed(seed)
            print(f"\n===== {model_name} | fold {fold} | seed {seed} =====")
            try:
                if model_name in ("catboost_raw", "xgboost_raw", "mlp_baseline"):
                    fold_of = pool_df["match_id"].map(assign)
                    tr_df = pool_df[fold_of != fold]
                    va_df = pool_df[fold_of == fold]
                    print(f"train={len(tr_df):,} | val={len(va_df):,} | test={len(test_df):,}")
                    if model_name == "catboost_raw":
                        out_val, out_test = run_catboost_raw(fold, tr_df, va_df, test_df,
                                                             args.gpu, args.iterations, seed)
                    elif model_name == "xgboost_raw":
                        out_val, out_test = run_xgboost_raw(fold, tr_df, va_df, test_df,
                                                            args.gpu, args.iterations, seed)
                    else:
                        out_val, out_test = run_mlp_baseline(fold, tr_df, va_df, test_df,
                                                             args.nn_epochs, seed)
                    del tr_df, va_df
                elif model_name in ("spadl_catboost", "spadl_xgboost"):
                    s_cat, s_cont = spadl_feature_cols(SPADL_CTX)
                    fold_of = spadl_pool["match_id"].map(assign)
                    tr_df = spadl_pool[fold_of != fold]
                    va_df = spadl_pool[fold_of == fold]
                    print(f"train={len(tr_df):,} | val={len(va_df):,} | test={len(spadl_test):,}")
                    runner = run_catboost_raw if model_name == "spadl_catboost" else run_xgboost_raw
                    out_val, out_test = runner(fold, tr_df, va_df, spadl_test,
                                               args.gpu, args.iterations, seed,
                                               cat_cols=s_cat, cont_cols=s_cont)
                    del tr_df, va_df
                elif model_name == "tt_catboost_pca":
                    fold_arr = np.vectorize(lambda m: assign.get(int(m), -1))(pca["mids"])
                    tr_mask, va_mask = fold_arr != fold, fold_arr == fold
                    print(f"train={tr_mask.sum():,} | val={va_mask.sum():,} | test={pca['test'][0].shape[0]:,}")
                    te_mids, te_aidx = emb_test_meta(pca["test"])
                    out_val, out_test = run_tt_catboost_pca(
                        fold, pca["emb"], pca["score"], pca["concede"], pca["mids"],
                        tr_mask, va_mask, pca["test"][0], pca["test"][1], pca["test"][2],
                        te_mids, te_aidx, args.gpu, args.iterations, seed)
                else:  # tt_mlp
                    tr_masks, va_masks = [], []
                    for mids in seg911["seg_mids"]:
                        fold_arr = np.vectorize(lambda m: assign.get(int(m), -1))(mids)
                        tr_masks.append(fold_arr != fold)
                        va_masks.append(fold_arr == fold)
                    te = seg911["test"]
                    te_mids, te_aidx = emb_test_meta(te)
                    print(f"train={sum(m.sum() for m in tr_masks):,} | val={sum(m.sum() for m in va_masks):,}")
                    out_val, out_test = run_tt_mlp(fold, seg911["segments"], tr_masks, va_masks,
                                                   te[0], te[1], te[2], te_mids, te_aidx,
                                                   args.nn_epochs, seed)

                results, pred_df = evaluate_both_tasks(out_val, out_test)
                write_cell(args.results_dir, model_name, fold, seed, results, pred_df)
                del out_val, out_test
                gc.collect()
            except Exception:
                err = traceback.format_exc()
                print(f"[ERROR] {model_name} fold{fold}:\n{err}")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(err)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    aggregate(args.results_dir)


if __name__ == "__main__":
    main()
