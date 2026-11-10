import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from train_xg_tabtransformer import (
    MAX_TEST_MATCH_FILES,
    PREDICTIONS_OUTPUT,
    TEST_ROOT,
    binary_f1_score,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    find_best_threshold,
    is_ball_out_event,
    is_definitive_turnover,
    is_open_play_event,
    safe_num,
    safe_text,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass

COMPARE_OUTPUT = "statsbomb_xg_comparison_metrics.csv"
JOINED_OUTPUT = "statsbomb_vs_model_xg_predictions.csv"
METRICS_MODEL_NAME = "tabtransformer_xg"
METRICS_BASELINE_NAME = "statsbomb_builtin_xg"

RAW_COMPARE_COLUMNS = [
    "match_id",
    "index",
    "type",
    "team",
    "play_pattern",
    "period",
    "minute",
    "second",
    "shot_outcome",
    "shot_statsbomb_xg",
    "pass_outcome",
    "interception_outcome",
    "out",
]

SKIP_TYPES = {"Starting XI", "Half Start"}


def safe_prob(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def extract_statsbomb_shots_from_match(match_events: pd.DataFrame, source_name: str) -> List[Dict[str, object]]:
    if len(match_events) == 0:
        return []

    df = match_events.sort_values(["period", "minute", "second", "index"]).reset_index(drop=True)
    match_id = int(df["match_id"].iloc[0]) if "match_id" in df.columns else -1

    rows: List[Dict[str, object]] = []
    current_team: Optional[str] = None
    opponent_touch_streak = 0
    last_period = None
    shot_idx = 0

    for _, row in df.iterrows():
        etype = safe_text(row.get("type"))
        team = safe_text(row.get("team"))
        period = int(safe_num(row.get("period")))

        if etype in SKIP_TYPES:
            continue

        if last_period is not None and period != last_period:
            current_team = None
            opponent_touch_streak = 0
        last_period = period

        if not is_open_play_event(row):
            current_team = None
            opponent_touch_streak = 0
            continue

        if current_team is None:
            current_team = team
            opponent_touch_streak = 0
        elif team != "__MISSING__" and current_team != "__MISSING__":
            if team == current_team:
                opponent_touch_streak = 0
            else:
                opponent_touch_streak += 1

        if etype == "Shot":
            rows.append(
                {
                    "match_id": match_id,
                    "source_name": source_name,
                    "sequence_id": shot_idx,
                    "label": 1 if safe_text(row.get("shot_outcome")) == "Goal" else 0,
                    "shot_outcome": safe_text(row.get("shot_outcome")),
                    "statsbomb_xg": safe_prob(row.get("shot_statsbomb_xg")),
                }
            )
            shot_idx += 1
            current_team = None
            opponent_touch_streak = 0
            continue

        if is_ball_out_event(row) or is_definitive_turnover(row) or opponent_touch_streak >= 3:
            current_team = None
            opponent_touch_streak = 0

    return rows


def load_statsbomb_shots_from_root(root_dir: str, max_match_files: Optional[int] = None) -> pd.DataFrame:
    event_files: List[Tuple[str, str]] = []
    for root, _, files in os.walk(root_dir):
        if os.path.basename(root) == "events":
            source_name = os.path.basename(os.path.dirname(root))
            for name in files:
                if name.endswith(".parquet"):
                    event_files.append((os.path.join(root, name), source_name))

    event_files = sorted(event_files)
    if max_match_files is not None:
        event_files = event_files[:max_match_files]

    rows: List[Dict[str, object]] = []
    for path, source_name in tqdm(event_files, desc=f"Reading {root_dir}"):
        try:
            df = pd.read_parquet(path, columns=RAW_COMPARE_COLUMNS)
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in RAW_COMPARE_COLUMNS if c in df.columns]
            df = df[keep].copy()
        rows.extend(extract_statsbomb_shots_from_match(df, source_name))

    return pd.DataFrame(rows)


def evaluate_probs(df: pd.DataFrame, prob_col: str, fixed_threshold: Optional[float] = None) -> Dict[str, float]:
    clean = df[["label", prob_col]].dropna().copy()
    labels = clean["label"].astype(int).tolist()
    probs = clean[prob_col].astype(float).clip(1e-7, 1.0 - 1e-7).tolist()

    if len(clean) == 0:
        return {
            "n": 0,
            "goal_rate": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "threshold": float("nan"),
        }

    threshold = fixed_threshold if fixed_threshold is not None else find_best_threshold(labels, probs)[0]
    preds = (np.asarray(probs) >= threshold).astype(int)
    return {
        "n": int(len(clean)),
        "goal_rate": float(clean["label"].mean()),
        "f1": float(binary_f1_score(labels, preds.tolist())),
        "roc_auc": float(binary_roc_auc(labels, probs)),
        "log_loss": float(binary_log_loss(labels, probs)),
        "brier": float(brier_score(labels, probs)),
        "threshold": float(threshold),
    }


def get_model_threshold(default: float = 0.5) -> float:
    metrics_path = "xg_tabtransformer_metrics.csv"
    if not os.path.exists(metrics_path):
        return default
    try:
        df = pd.read_csv(metrics_path)
        overall = df[df["split"] == "overall_test"]
        if len(overall) > 0 and "threshold" in overall.columns:
            return float(overall.iloc[0]["threshold"])
    except Exception:
        pass
    return default


def print_metric_line(title: str, metrics: Dict[str, float]) -> None:
    print(
        f"{title} | n={metrics['n']} | goal_rate={metrics['goal_rate']:.4f} | "
        f"F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f} | Brier={metrics['brier']:.4f} | "
        f"threshold={metrics['threshold']:.2f}"
    )


def main() -> None:
    statsbomb_df = load_statsbomb_shots_from_root(TEST_ROOT, max_match_files=MAX_TEST_MATCH_FILES)
    if len(statsbomb_df) == 0:
        raise RuntimeError("No open-play shots were extracted from dataset-test.")

    valid_statsbomb_df = statsbomb_df.dropna(subset=["statsbomb_xg"]).copy()
    print(f"Extracted held-out shots: {len(statsbomb_df)} | with StatsBomb xG: {len(valid_statsbomb_df)}")

    statsbomb_metrics = evaluate_probs(valid_statsbomb_df, "statsbomb_xg", fixed_threshold=0.5)
    print_metric_line("StatsBomb built-in xG (threshold=0.50)", statsbomb_metrics)

    comparison_rows: List[Dict[str, object]] = [
        {
            "model": METRICS_BASELINE_NAME,
            "split": "overall_test",
            "source_name": "ALL",
            **statsbomb_metrics,
        }
    ]

    for source_name, group in valid_statsbomb_df.groupby("source_name"):
        group_metrics = evaluate_probs(group, "statsbomb_xg", fixed_threshold=0.5)
        comparison_rows.append(
            {
                "model": METRICS_BASELINE_NAME,
                "split": "heldout_tournament",
                "source_name": source_name,
                **group_metrics,
            }
        )
        print_metric_line(f"{source_name} | StatsBomb built-in xG", group_metrics)

    if os.path.exists(PREDICTIONS_OUTPUT):
        model_df = pd.read_csv(PREDICTIONS_OUTPUT)
        join_cols = ["match_id", "source_name", "sequence_id", "label"]
        merged = pd.merge(model_df, valid_statsbomb_df, on=join_cols, how="inner")
        merged.to_csv(JOINED_OUTPUT, index=False)

        model_threshold = get_model_threshold(default=0.5)
        model_metrics = evaluate_probs(merged, "goal_prob", fixed_threshold=model_threshold)
        baseline_on_same_shots = evaluate_probs(merged, "statsbomb_xg", fixed_threshold=0.5)

        comparison_rows.extend(
            [
                {
                    "model": METRICS_MODEL_NAME,
                    "split": "overall_test_intersection",
                    "source_name": "ALL",
                    **model_metrics,
                },
                {
                    "model": f"{METRICS_BASELINE_NAME}_intersection",
                    "split": "overall_test_intersection",
                    "source_name": "ALL",
                    **baseline_on_same_shots,
                },
            ]
        )

        print("\n=== Same-shot comparison (intersection with your model predictions) ===")
        print_metric_line(f"TabTransformer xG (threshold={model_threshold:.2f})", model_metrics)
        print_metric_line("StatsBomb built-in xG (threshold=0.50)", baseline_on_same_shots)

        for source_name, group in merged.groupby("source_name"):
            group_model = evaluate_probs(group, "goal_prob", fixed_threshold=model_threshold)
            group_statsbomb = evaluate_probs(group, "statsbomb_xg", fixed_threshold=0.5)
            comparison_rows.append(
                {
                    "model": METRICS_MODEL_NAME,
                    "split": "heldout_tournament_intersection",
                    "source_name": source_name,
                    **group_model,
                }
            )
            comparison_rows.append(
                {
                    "model": f"{METRICS_BASELINE_NAME}_intersection",
                    "split": "heldout_tournament_intersection",
                    "source_name": source_name,
                    **group_statsbomb,
                }
            )
            print_metric_line(f"{source_name} | TabTransformer xG", group_model)
            print_metric_line(f"{source_name} | StatsBomb built-in xG", group_statsbomb)

        print(f"\nJoined predictions saved to: {JOINED_OUTPUT}")
    else:
        print(f"Model predictions file not found: {PREDICTIONS_OUTPUT}")
        print("Only StatsBomb built-in xG metrics were computed.")

    metrics_df = pd.DataFrame(comparison_rows)
    metrics_df.to_csv(COMPARE_OUTPUT, index=False)
    print(f"Comparison metrics saved to: {COMPARE_OUTPUT}")


if __name__ == "__main__":
    main()
