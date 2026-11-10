"""
train_vaep_spadl_baseline.py

用 SPADL 特征 + CatBoost 复现 Decroos 2019 VAEP 方法。

SPADL (Soccer Player Action Description Language) 将 StatsBomb 事件简化为：
  - type_name   : 简化动作类型（~22 种，如 pass/shot/cross/take_on 等）
  - result_name : 结果（success/fail/offside/yellow_card/red_card）
  - bodypart_name: 身体部位（foot/head/other/no_touch）
  - x, y        : 起始坐标（StatsBomb 120×80 归一化 → 0~1）
  - end_x, end_y: 终点坐标
  - period      : 比赛阶段
  - time_seconds: 阶段内时间（秒）

论文设计：拼接当前动作 + 前 k=2 个动作的特征（共 3×9=27 维）。
模型：CatBoost（与 train_vaep_catboost.py 保持一致，方便特征价值对比）。

与其他 VAEP 脚本对比：
  - train_vaep_baseline.py       : StatsBomb 44 特征 + 自训 embedding MLP
  - train_vaep_tabtransformer.py : StatsBomb 44 特征 + 冻结预训练 encoder
  - train_vaep_catboost.py       : StatsBomb 44 特征 + CatBoost  ← 相同模型
  - train_vaep_xgboost.py        : StatsBomb 44 特征 + XGBoost
  - train_vaep_spadl_baseline.py : SPADL 27 特征 + CatBoost     ← 本脚本

用法：
  python train_vaep_spadl_baseline.py
  python train_vaep_spadl_baseline.py --gpu
  python train_vaep_spadl_baseline.py --context-size 0   # 只用当前动作
  python train_vaep_spadl_baseline.py --eval-only
"""

import argparse
import gc
import hashlib
import os
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
    choose_prediction_output_path,
    find_best_threshold,
    list_event_files,
    safe_num,
    safe_text,
    save_predictions_compact,
    summarize_split,
)
from train_vaep_catboost import _split_files_train_val
from train_xg_tabtransformer import set_seed

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

# ---------------------------------------------------------------------------
CHECKPOINT_DIR     = "checkpoints"
SCORE_MODEL_PATH   = os.path.join(CHECKPOINT_DIR, "vaep_spadl_score.cbm")
CONCEDE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vaep_spadl_concede.cbm")
PREDICTIONS_OUTPUT = "vaep_spadl_predictions"
METRICS_OUTPUT     = "vaep_spadl_metrics.csv"

LOOKAHEAD_K       = 10
CONTEXT_SIZE      = 2   # 前驱动作数 k（论文 k=2，即 3 个动作拼接）
ITERATIONS        = 1000
DEPTH             = 6
LEARNING_RATE     = 0.1
L2_LEAF_REG       = 3.0
EARLY_STOPPING_ROUNDS = 50

# SPADL 每个动作的基础特征列
SPADL_CAT  = ["type_name", "result_name", "bodypart_name"]
SPADL_CONT = ["x", "y", "end_x", "end_y", "period", "time_seconds"]

# 不映射为 SPADL 动作的 StatsBomb 事件类型
_SKIP_TYPES = {
    "Starting XI", "Half Start", "Half End", "Substitution",
    "Tactical Shift", "Player On", "Player Off",
    "Camera On", "Camera off", "50/50", "Pressure",
    "Ball Receipt*", "Error", "Referee Ball-Drop",
    "Shield", "Injury Stoppage", "Offside",
}

# 序列上下文的 padding 值
_PAD_CAT  = {c: "__PAD__" for c in SPADL_CAT}
_PAD_CONT = {c: 0.0 for c in SPADL_CONT}
_PAD_FEAT = {**_PAD_CAT, **_PAD_CONT}


# ---------------------------------------------------------------------------
# SPADL 转换工具
# ---------------------------------------------------------------------------
def _xy(row, key: str) -> Tuple[float, float]:
    loc = row.get(key)
    if loc is None:
        return 0.0, 0.0
    try:
        loc = list(loc)
    except TypeError:
        return 0.0, 0.0
    x = float(loc[0]) if len(loc) > 0 else 0.0
    y = float(loc[1]) if len(loc) > 1 else 0.0
    return x, y


def _is_goal(row) -> bool:
    return (safe_text(row.get("type")) == "Shot" and
            safe_text(row.get("shot_outcome")) == "Goal")


def map_to_spadl(row) -> Optional[Dict]:
    """StatsBomb 事件行 → SPADL 特征 dict。
    不可映射的事件返回 None。
    坐标归一化：StatsBomb 120×80 → 0~1。
    """
    etype = safe_text(row.get("type"))
    if etype in _SKIP_TYPES:
        return None

    # --- 坐标 ---
    x, y = _xy(row, "location")
    x /= 120.0; y /= 80.0

    if etype == "Pass":
        ex, ey = _xy(row, "pass_end_location")
    elif etype == "Carry":
        ex, ey = _xy(row, "carry_end_location")
    elif etype == "Shot":
        ex, ey = _xy(row, "shot_end_location")
    else:
        ex, ey = x, y  # 无终点信息则与起点相同
    ex /= 120.0; ey /= 80.0

    period = float(row.get("period") or 1)
    minute = int(row.get("minute") or 0)
    second = int(row.get("second") or 0)
    time_seconds = float(minute * 60 + second)

    # --- 动作类型 ---
    if etype == "Pass":
        cross = str(row.get("pass_cross", "")).lower() in ("true", "1")
        tech  = safe_text(row.get("pass_technique"))
        ptype = safe_text(row.get("pass_type"))
        if tech == "Throw-in":
            type_name = "throw_in"
        elif "Corner" in ptype:
            type_name = "corner_crossed" if cross else "corner_short"
        elif "Free Kick" in ptype:
            type_name = "freekick_crossed" if cross else "freekick_short"
        elif "Goal Kick" in ptype:
            type_name = "goalkick"
        else:
            type_name = "cross" if cross else "pass"

    elif etype == "Shot":
        stype = safe_text(row.get("shot_type"))
        if stype == "Penalty":
            type_name = "shot_penalty"
        elif stype == "Free Kick":
            type_name = "shot_freekick"
        else:
            type_name = "shot"

    elif etype == "Dribble":
        type_name = "take_on"
    elif etype == "Carry":
        type_name = "dribble"
    elif etype == "Clearance":
        type_name = "clearance"
    elif etype == "Interception":
        type_name = "interception"
    elif etype in ("Foul Committed", "Bad Behaviour"):
        type_name = "foul"
    elif etype == "Duel":
        duel_type = safe_text(row.get("duel_type"))
        if "Tackle" in duel_type:
            type_name = "tackle"
        else:
            return None  # 空中争顶等 → 跳过
    elif etype == "Goal Keeper":
        gkt = safe_text(row.get("goalkeeper_type"))
        if "Save" in gkt:
            type_name = "keeper_save"
        elif "Claim" in gkt:
            type_name = "keeper_claim"
        elif "Punch" in gkt:
            type_name = "keeper_punch"
        elif "Pick Up" in gkt or "Collect" in gkt:
            type_name = "keeper_pick_up"
        else:
            type_name = "keeper_save"
    elif etype == "Block":
        type_name = "tackle"
    elif etype == "Ball Recovery":
        type_name = "interception"
    elif etype == "Miscontrol":
        type_name = "bad_touch"
    else:
        type_name = "non_action"

    # --- 结果 ---
    if etype == "Pass":
        po = safe_text(row.get("pass_outcome"))
        if po in ("__MISSING__", "Complete", "Unknown"):
            result_name = "success"
        elif po == "Offside":
            result_name = "offside"
        else:
            result_name = "fail"
    elif etype == "Shot":
        so = safe_text(row.get("shot_outcome"))
        if so == "Goal":
            result_name = "success"
        elif so == "Offside":
            result_name = "offside"
        else:
            result_name = "fail"
    elif etype == "Dribble":
        result_name = ("success"
                       if safe_text(row.get("dribble_outcome")) == "Complete"
                       else "fail")
    elif etype in ("Foul Committed", "Bad Behaviour"):
        ft = safe_text(row.get("foul_committed_type"))
        if "Yellow" in ft:
            result_name = "yellow_card"
        elif "Red" in ft:
            result_name = "red_card"
        else:
            result_name = "success"
    else:
        result_name = "success"

    # --- 身体部位 ---
    if etype == "Pass":
        bp = safe_text(row.get("pass_body_part"))
    elif etype == "Shot":
        bp = safe_text(row.get("shot_body_part"))
    else:
        bp = "__MISSING__"

    if "Foot" in bp:
        bodypart_name = "foot"
    elif "Head" in bp:
        bodypart_name = "head"
    elif bp == "__MISSING__":
        bodypart_name = "no_touch"
    else:
        bodypart_name = "other"

    return {
        "type_name":     sys.intern(type_name),
        "result_name":   sys.intern(result_name),
        "bodypart_name": sys.intern(bodypart_name),
        "x": x, "y": y, "end_x": ex, "end_y": ey,
        "period": period, "time_seconds": time_seconds,
    }


# ---------------------------------------------------------------------------
# 逐比赛构建 SPADL 动作序列（含标签 + 序列上下文）
# ---------------------------------------------------------------------------
def build_spadl_actions_from_match(
    df: pd.DataFrame,
    source_name: str,
    lookahead_k: int,
    context_size: int,
) -> List[Dict]:
    sort_cols = [c for c in ["period", "minute", "second", "index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    match_id = int(df["match_id"].iloc[0]) if "match_id" in df.columns else -1

    # Pass 1：提取 SPADL 特征
    actions = []
    for idx, row in df.iterrows():
        spadl = map_to_spadl(row)
        if spadl is None:
            continue
        actions.append({
            "match_id":     match_id,
            "source_name":  source_name,
            "action_index": int(idx),
            "team":         safe_text(row.get("team")),
            "is_goal":      1 if _is_goal(row) else 0,
            "spadl":        spadl,
        })

    if not actions:
        return []

    # Pass 2：VAEP 标签（前向 lookahead_k 个动作）
    teams = [a["team"] for a in actions]
    goals = [a["is_goal"] for a in actions]

    for i, action in enumerate(actions):
        team_i = teams[i]
        score_label = 0
        concede_label = 0
        upper = min(len(actions), i + lookahead_k + 1)
        for j in range(i + 1, upper):
            if goals[j]:
                if teams[j] == team_i:
                    score_label = 1
                else:
                    concede_label = 1
                if score_label and concede_label:
                    break
        action["score_label"]   = score_label
        action["concede_label"] = concede_label

    # Pass 3：拼接序列上下文（当前 + 前 context_size 个动作）
    for i, action in enumerate(actions):
        flat_feat: Dict[str, object] = {}
        for t, ctx_i in enumerate(range(i - context_size, i + 1)):
            feat = actions[ctx_i]["spadl"] if ctx_i >= 0 else _PAD_FEAT
            for col in SPADL_CAT:
                flat_feat[f"{col}_{t}"] = feat[col]
            for col in SPADL_CONT:
                flat_feat[f"{col}_{t}"] = float(feat[col])
        action["flat_feat"] = flat_feat

    return actions


# ---------------------------------------------------------------------------
# 流式构建 DataFrame + 缓存
# ---------------------------------------------------------------------------
def _get_feature_cols(context_size: int) -> Tuple[List[str], List[str]]:
    cat_cols  = [f"{c}_{t}" for t in range(context_size + 1) for c in SPADL_CAT]
    cont_cols = [f"{c}_{t}" for t in range(context_size + 1) for c in SPADL_CONT]
    return cat_cols, cont_cols


def build_spadl_df_streaming(
    files: list,
    lookahead_k: int,
    context_size: int,
    desc: str = "Building",
) -> pd.DataFrame:
    cat_cols, cont_cols = _get_feature_cols(context_size)
    cat_data  = {col: [] for col in cat_cols}
    cont_data = {col: [] for col in cont_cols}
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

        actions = build_spadl_actions_from_match(
            df_raw, source_name, lookahead_k, context_size
        )
        del df_raw

        for a in actions:
            ff = a["flat_feat"]
            for col in cat_cols:
                cat_data[col].append(ff.get(col, "__PAD__"))
            for col in cont_cols:
                cont_data[col].append(float(ff.get(col, 0.0)))
            score_labels.append(int(a["score_label"]))
            concede_labels.append(int(a["concede_label"]))
            match_ids.append(int(a["match_id"]))
            source_names.append(sys.intern(source_name))
            action_indices.append(int(a["action_index"]))
        del actions

    if not score_labels:
        return pd.DataFrame(columns=cat_cols + cont_cols +
                            ["score_label", "concede_label", "match_id",
                             "source_name", "action_index"])

    data: Dict = {}
    for col in cat_cols:
        data[col] = cat_data[col]
    for col in cont_cols:
        data[col] = np.array(cont_data[col], dtype=np.float32)
    data["score_label"]   = np.array(score_labels,   dtype=np.int8)
    data["concede_label"] = np.array(concede_labels, dtype=np.int8)
    data["match_id"]      = np.array(match_ids,      dtype=np.int32)
    data["source_name"]   = source_names
    data["action_index"]  = np.array(action_indices, dtype=np.int32)
    return pd.DataFrame(data)


def load_or_build_spadl_df(
    name: str,
    files: list,
    lookahead_k: int,
    context_size: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key_src = "|".join(sorted(p for p, _ in files)) + f"_k{lookahead_k}_ctx{context_size}"
    key = hashlib.md5(key_src.encode()).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"spadl_{name}_{key}.parquet")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached {name} SPADL df from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Building {name} SPADL df from {len(files)} match files...")
    df = build_spadl_df_streaming(files, lookahead_k, context_size, desc=f"  {name}")

    if use_cache and len(df) > 0:
        print(f"Caching {name} SPADL df → {cache_path}")
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            print(f"Warning: could not cache ({exc})")
    return df


# ---------------------------------------------------------------------------
# CatBoost 工具
# ---------------------------------------------------------------------------
def make_pool(df: pd.DataFrame, cat_cols: List[str], cont_cols: List[str],
              label_col: str) -> Pool:
    X = df[cat_cols + cont_cols]
    y = df[label_col].values
    return Pool(X, label=y, cat_features=list(range(len(cat_cols))))


def predict_proba(model: CatBoostClassifier, df: pd.DataFrame,
                  cat_cols: List[str], cont_cols: List[str],
                  label_col: str, prob_col: str) -> pd.DataFrame:
    probs = model.predict_proba(df[cat_cols + cont_cols])[:, 1]
    return pd.DataFrame({
        "match_id":     df["match_id"].values,
        "source_name":  df["source_name"].values,
        "action_index": df["action_index"].values,
        label_col:      df[label_col].values,
        prob_col:       probs,
    })


# ---------------------------------------------------------------------------
# 结果导出
# ---------------------------------------------------------------------------
def _export(args, valid_splits, train_dfs, val_dfs, test_dfs):
    def _merge(score_df, concede_df):
        if len(score_df) == 0:
            return pd.DataFrame()
        return score_df.merge(
            concede_df[["match_id", "action_index", "concede_label", "concede_prob"]],
            on=["match_id", "action_index"],
            how="inner",
        )

    val_pred   = _merge(*val_dfs)
    test_pred  = _merge(*test_dfs)
    train_pred = _merge(*train_dfs) if train_dfs else pd.DataFrame()

    if len(val_pred) > 0:
        score_thr,   _ = find_best_threshold(
            val_pred["score_label"].astype(int).tolist(),
            val_pred["score_prob"].astype(float).tolist())
        concede_thr, _ = find_best_threshold(
            val_pred["concede_label"].astype(int).tolist(),
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
    for sname, df_s in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        if len(df_s) > 0:
            metrics_rows.extend(summarize_split(sname, df_s, score_thr, concede_thr))
    pd.DataFrame(metrics_rows).to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\nSaved predictions : {output_path}")
    print(f"Saved metrics     : {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")

    if len(val_pred) > 0:
        s_ll = binary_log_loss(val_pred["score_label"].tolist(), val_pred["score_prob"].tolist())
        c_ll = binary_log_loss(val_pred["concede_label"].tolist(), val_pred["concede_prob"].tolist())
        print(f"Val score_log_loss={s_ll:.4f} | concede_log_loss={c_ll:.4f}")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAEP with SPADL features + CatBoost")
    p.add_argument("--context-size",    type=int, default=CONTEXT_SIZE,
                   help="Number of preceding actions to concatenate (paper=2)")
    p.add_argument("--iterations",      type=int, default=ITERATIONS)
    p.add_argument("--depth",           type=int, default=DEPTH)
    p.add_argument("--lr",              type=float, default=LEARNING_RATE)
    p.add_argument("--early-stopping",  type=int, default=EARLY_STOPPING_ROUNDS)
    p.add_argument("--eval-only",       action="store_true")
    p.add_argument("--prediction-splits",  default="val,test")
    p.add_argument("--prediction-format",  default="csv_gz",
                   choices=["csv", "csv_gz", "parquet"])
    p.add_argument("--include-source-name", action="store_true")
    p.add_argument("--max-train-matches", type=int, default=None)
    p.add_argument("--max-test-matches",  type=int, default=None)
    p.add_argument("--gpu",             action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(SEED)

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ctx  = args.context_size
    cat_cols, cont_cols = _get_feature_cols(ctx)
    print(f"SPADL feature dims: {len(cat_cols)} cat + {len(cont_cols)} cont "
          f"(context_size={ctx}, total actions={ctx+1})")

    max_train = args.max_train_matches or MAX_TRAIN_MATCH_FILES
    max_test  = args.max_test_matches  or MAX_TEST_MATCH_FILES
    train_files_all = list_event_files(TRAIN_ROOT, max_match_files=max_train)
    train_files, val_files = _split_files_train_val(train_files_all, VAL_RATIO, SEED)
    test_files = list_event_files(TEST_ROOT, max_match_files=max_test)
    print(f"Files — train={len(train_files)} | val={len(val_files)} | test={len(test_files)}")

    task_type = "GPU" if args.gpu else "CPU"

    def _train_model(train_pool, val_pool, label) -> CatBoostClassifier:
        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.lr,
            l2_leaf_reg=L2_LEAF_REG,
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

    # ------------------------------------------------------------------
    # Eval-only fast path
    # ------------------------------------------------------------------
    if args.eval_only:
        if not os.path.exists(SCORE_MODEL_PATH) or not os.path.exists(CONCEDE_MODEL_PATH):
            raise FileNotFoundError("Models not found. Run training first.")
        score_model   = CatBoostClassifier(); score_model.load_model(SCORE_MODEL_PATH)
        concede_model = CatBoostClassifier(); concede_model.load_model(CONCEDE_MODEL_PATH)

        def _pred_files(name, files):
            if not files:
                return pd.DataFrame(), pd.DataFrame()
            df = load_or_build_spadl_df(name, files, LOOKAHEAD_K, ctx, use_cache=True)
            s = predict_proba(score_model,   df, cat_cols, cont_cols, "score_label",   "score_prob")
            c = predict_proba(concede_model, df, cat_cols, cont_cols, "concede_label", "concede_prob")
            del df; gc.collect()
            return s, c

        train_dfs = _pred_files("train", train_files) if "train" in valid_splits else None
        val_dfs   = _pred_files("val",   val_files)
        test_dfs  = _pred_files("test",  test_files)
        _export(args, valid_splits, train_dfs, val_dfs, test_dfs)
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    print("\nLoading train split...")
    train_df = load_or_build_spadl_df("train", train_files, LOOKAHEAD_K, ctx, use_cache=True)
    print(f"  train rows: {len(train_df):,}")

    print("\nLoading val split...")
    val_df = load_or_build_spadl_df("val", val_files, LOOKAHEAD_K, ctx, use_cache=True)
    print(f"  val rows:   {len(val_df):,}")

    val_score_pool   = make_pool(val_df, cat_cols, cont_cols, "score_label")
    val_concede_pool = make_pool(val_df, cat_cols, cont_cols, "concede_label")

    # Train score
    train_score_pool = make_pool(train_df, cat_cols, cont_cols, "score_label")
    score_model = _train_model(train_score_pool, val_score_pool, "score")
    del train_score_pool; gc.collect()

    # Train concede
    train_concede_pool = make_pool(train_df, cat_cols, cont_cols, "concede_label")
    concede_model = _train_model(train_concede_pool, val_concede_pool, "concede")
    del train_concede_pool, val_score_pool, val_concede_pool; gc.collect()

    # Save
    score_model.save_model(SCORE_MODEL_PATH)
    concede_model.save_model(CONCEDE_MODEL_PATH)
    print(f"\nSaved: {SCORE_MODEL_PATH}")
    print(f"Saved: {CONCEDE_MODEL_PATH}")

    # Predict
    def _pred(df):
        s = predict_proba(score_model,   df, cat_cols, cont_cols, "score_label",   "score_prob")
        c = predict_proba(concede_model, df, cat_cols, cont_cols, "concede_label", "concede_prob")
        return s, c

    train_dfs = _pred(train_df) if "train" in valid_splits else (pd.DataFrame(), pd.DataFrame())
    del train_df; gc.collect()

    val_dfs = _pred(val_df)
    del val_df; gc.collect()

    print("\nLoading test split...")
    test_df = load_or_build_spadl_df("test", test_files, LOOKAHEAD_K, ctx, use_cache=True)
    print(f"  test rows:  {len(test_df):,}")
    test_dfs = _pred(test_df)
    del test_df; gc.collect()

    _export(args, valid_splits, train_dfs, val_dfs, test_dfs)


if __name__ == "__main__":
    main()
