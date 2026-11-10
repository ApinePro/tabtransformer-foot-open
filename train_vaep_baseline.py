import argparse
import hashlib
import math
import os
import pickle
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    DEFAULT_CAT_COLS,
    DEFAULT_CONT_COLS,
    RAW_EVENT_COLUMNS,
    TEST_ROOT,
    TRAIN_ROOT,
    VAL_RATIO,
    WEIGHT_MODE,
    MAX_POS_WEIGHT,
    binary_f1_score,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    build_event_features,
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

MAX_TRAIN_MATCH_FILES: Optional[int] = None
MAX_TEST_MATCH_FILES: Optional[int] = None

LOOKAHEAD_ACTIONS = 10
BATCH_SIZE = 256
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.2
USE_CLASS_WEIGHTS = True

CHECKPOINT_OUTPUT = os.path.join("checkpoints", with_exp_suffix("vaep_baseline_best.pt"))
METRICS_OUTPUT = with_exp_suffix("vaep_baseline_metrics.csv")
PREDICTIONS_OUTPUT = with_exp_suffix("vaep_baseline_predictions.csv")
CACHE_DIR = os.path.join(".", ".vaep_cache")
STATS_CHUNK_SIZE = 50000

SKIP_TYPES = {"Starting XI", "Half Start"}


def safe_num(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def is_goal_event(row: pd.Series) -> bool:
    return safe_text(row.get("type")) == "Shot" and safe_text(row.get("shot_outcome")) == "Goal"


def list_event_files(root_dir: str, max_match_files: Optional[int] = None) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for root, _, names in os.walk(root_dir):
        if os.path.basename(root) != "events":
            continue
        source_name = os.path.basename(os.path.dirname(root))
        for name in names:
            if name.endswith(".parquet"):
                files.append((os.path.join(root, name), source_name))

    files = sorted(files)
    if max_match_files is not None:
        files = files[:max_match_files]
    return files


def build_actions_from_match(df: pd.DataFrame, source_name: str, lookahead_k: int) -> List[Dict[str, object]]:
    if len(df) == 0:
        return []

    sort_cols = [c for c in ["period", "minute", "second", "index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    match_id = int(df["match_id"].iloc[0]) if "match_id" in df.columns else -1
    records: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        etype = safe_text(row.get("type"))
        if etype in SKIP_TYPES:
            continue

        team = safe_text(row.get("team"))
        possession_team = safe_text(row.get("possession_team"))
        sequence_team = possession_team if possession_team != "__MISSING__" else team
        feat = build_event_features(row, DEFAULT_CAT_COLS, DEFAULT_CONT_COLS, sequence_team=sequence_team)

        records.append(
            {
                "match_id": match_id,
                "source_name": source_name,
                "action_index": int(idx),
                "team": team,
                "is_goal_event": 1 if is_goal_event(row) else 0,
                "features": feat,
            }
        )

    if not records:
        return records

    teams = [str(r["team"]) for r in records]
    goals = [int(r["is_goal_event"]) for r in records]

    for i, record in enumerate(records):
        team_i = teams[i]
        score_label = 0
        concede_label = 0
        upper = min(len(records), i + lookahead_k + 1)

        for j in range(i + 1, upper):
            if goals[j] != 1:
                continue
            if teams[j] == team_i:
                score_label = 1
            else:
                concede_label = 1
            if score_label == 1 and concede_label == 1:
                break

        record["score_label"] = score_label
        record["concede_label"] = concede_label

    return records


def load_action_dataset(root_dir: str, lookahead_k: int, max_match_files: Optional[int] = None) -> List[Dict[str, object]]:
    all_actions: List[Dict[str, object]] = []
    for path, source_name in tqdm(list_event_files(root_dir, max_match_files=max_match_files), desc=f"Loading {root_dir}"):
        try:
            df = pd.read_parquet(path, columns=RAW_EVENT_COLUMNS)
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in RAW_EVENT_COLUMNS if c in df.columns]
            df = df[keep].copy()

        all_actions.extend(build_actions_from_match(df, source_name, lookahead_k=lookahead_k))
    return all_actions


def get_cache_key(root_dir: str, lookahead_k: int, max_match_files: Optional[int]) -> str:
    key_str = f"{root_dir}_{lookahead_k}_{max_match_files}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_or_build_actions(
    root_dir: str,
    lookahead_k: int,
    max_match_files: Optional[int] = None,
    use_cache: bool = True,
) -> List[Dict[str, object]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_cache_key(root_dir, lookahead_k, max_match_files)
    cache_path = os.path.join(CACHE_DIR, f"actions_{cache_key}.pkl")
    
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached actions from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print(f"Building actions from {root_dir}...")
    actions = load_action_dataset(root_dir, lookahead_k, max_match_files)
    
    print(f"Caching actions to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(actions, f)
    
    return actions


def load_or_build_split_actions(
    root_dir: str,
    lookahead_k: int,
    val_ratio: float,
    seed: int,
    max_match_files: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return (train_actions, val_actions), caching the split result separately."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base_key = get_cache_key(root_dir, lookahead_k, max_match_files)
    split_key = hashlib.md5(f"{base_key}_val{val_ratio}_seed{seed}".encode()).hexdigest()
    train_cache = os.path.join(CACHE_DIR, f"train_split_{split_key}.pkl")
    val_cache = os.path.join(CACHE_DIR, f"val_split_{split_key}.pkl")

    if use_cache and os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Loading cached train split from {train_cache}...")
        with open(train_cache, "rb") as f:
            train_actions = pickle.load(f)
        print(f"Loading cached val split from {val_cache}...")
        with open(val_cache, "rb") as f:
            val_actions = pickle.load(f)
        return train_actions, val_actions

    train_all = load_or_build_actions(root_dir, lookahead_k, max_match_files, use_cache=use_cache)
    train_actions, val_actions = split_train_val_by_match(train_all, val_ratio=val_ratio, seed=seed)
    train_actions, val_actions = ensure_non_empty_split(train_actions, val_actions, seed=seed)
    del train_all

    if use_cache:
        for cache_path, data, label in [
            (train_cache, train_actions, "train split"),
            (val_cache, val_actions, "val split"),
        ]:
            print(f"Caching {label} to {cache_path}...")
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)
            except (OSError, MemoryError) as exc:
                print(f"Warning: could not cache {label} ({exc}). Continuing without cache.")
                # Remove partial file if it exists
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    return train_actions, val_actions


def build_cat_maps(
    actions: List[Dict[str, object]],
    cat_cols: List[str],
    chunk_size: int = STATS_CHUNK_SIZE,
) -> Tuple[Dict[str, Dict[str, int]], List[int]]:
    unique_sets: Dict[str, set] = {col: set() for col in cat_cols}
    for start in tqdm(range(0, len(actions), chunk_size), desc="Build cat maps"):
        chunk = actions[start : start + chunk_size]
        for action in chunk:
            feat = action["features"]
            for col in cat_cols:
                unique_sets[col].add(safe_text(feat.get(col)))

    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        uniques = sorted(unique_sets[col])
        cat_maps[col] = {value: idx + 1 for idx, value in enumerate(uniques)}
    cat_dims = [len(cat_maps[col]) + 1 for col in cat_cols]
    return cat_maps, cat_dims


def compute_cont_stats(
    actions: List[Dict[str, object]],
    cont_cols: List[str],
    chunk_size: int = STATS_CHUNK_SIZE,
) -> np.ndarray:
    if len(actions) == 0:
        return np.asarray([(0.0, 1.0) for _ in cont_cols], dtype=np.float32)

    num_cols = len(cont_cols)
    count = np.zeros(num_cols, dtype=np.float64)
    mean = np.zeros(num_cols, dtype=np.float64)
    m2 = np.zeros(num_cols, dtype=np.float64)

    for start in tqdm(range(0, len(actions), chunk_size), desc="Cont stats chunks"):
        chunk = actions[start : start + chunk_size]
        if not chunk:
            continue

        chunk_arr = np.asarray(
            [[safe_num(action["features"].get(col)) for col in cont_cols] for action in chunk],
            dtype=np.float64,
        )
        n_chunk = float(chunk_arr.shape[0])
        chunk_mean = chunk_arr.mean(axis=0)
        chunk_m2 = ((chunk_arr - chunk_mean) ** 2).sum(axis=0)

        total = count + n_chunk
        delta = chunk_mean - mean
        mean = mean + delta * (n_chunk / np.maximum(total, 1.0))
        m2 = m2 + chunk_m2 + (delta**2) * (count * n_chunk / np.maximum(total, 1.0))
        count = total

    variance = m2 / np.maximum(count, 1.0)
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1e-6] = 1.0

    return np.asarray([(float(mean[i]), float(std[i])) for i in range(num_cols)], dtype=np.float32)

class VAEPTabularDataset(Dataset):
    def __init__(
        self,
        action_indices: List[int],
        actions_source: List[Dict[str, object]],
        cat_cols: List[str],
        cont_cols: List[str],
        cat_maps: Dict[str, Dict[str, int]],
    ):
        self.action_indices = action_indices
        self.actions_source = actions_source
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.cat_maps = cat_maps

    def __len__(self) -> int:
        return len(self.action_indices)

    def __getitem__(self, idx: int):
        action_id = self.action_indices[idx]
        row = self.actions_source[action_id]
        feat = row["features"]

        cat_values = [self.cat_maps[col].get(safe_text(feat.get(col)), 0) for col in self.cat_cols]
        cont_values = [safe_num(feat.get(col)) for col in self.cont_cols]

        return (
            torch.tensor(cat_values, dtype=torch.long),
            torch.tensor(cont_values, dtype=torch.float32),
            torch.tensor(float(row["score_label"]), dtype=torch.float32),
            torch.tensor(float(row["concede_label"]), dtype=torch.float32),
            int(row["match_id"]),
            str(row["source_name"]),
            int(row["action_index"]),
        )


class VAEPMultitaskMLP(nn.Module):
    def __init__(self, cat_dims: List[int], num_cont: int, cont_mean_std: np.ndarray):
        super().__init__()
        emb_dims = [max(4, min(32, int(round(1.6 * (dim ** 0.56))))) for dim in cat_dims]
        self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim, emb_dim in zip(cat_dims, emb_dims)])

        mean_std = torch.tensor(cont_mean_std, dtype=torch.float32)
        self.register_buffer("cont_mean", mean_std[:, 0])
        self.register_buffer("cont_std", mean_std[:, 1])

        in_dim = int(sum(emb_dims) + num_cont)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.score_head = nn.Linear(HIDDEN_DIM // 2, 1)
        self.concede_head = nn.Linear(HIDDEN_DIM // 2, 1)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = [emb(x_categ[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = torch.cat(embs, dim=-1) if embs else torch.empty((x_categ.shape[0], 0), device=x_categ.device)

        x_cont_norm = (x_cont - self.cont_mean) / self.cont_std
        x = torch.cat([x_cat, x_cont_norm], dim=-1)
        z = self.backbone(x)
        return self.score_head(z).squeeze(-1), self.concede_head(z).squeeze(-1)


def compute_pos_weight(labels: List[int]) -> float:
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    if n_pos == 0 or not USE_CLASS_WEIGHTS:
        return 1.0

    ratio = n_neg / max(n_pos, 1)
    if WEIGHT_MODE == "ratio":
        pos_weight = ratio
    elif WEIGHT_MODE == "capped":
        pos_weight = min(MAX_POS_WEIGHT, ratio)
    else:
        pos_weight = min(MAX_POS_WEIGHT, math.sqrt(ratio))
    return float(pos_weight)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x_categ, x_cont, score_y, concede_y, *_ in tqdm(loader, desc="Train batches", leave=False):
        x_categ = x_categ.to(device, non_blocking=True)
        x_cont = x_cont.to(device, non_blocking=True)
        score_y = score_y.to(device, non_blocking=True)
        concede_y = concede_y.to(device, non_blocking=True)

        score_logits, concede_logits = model(x_categ, x_cont)
        loss_score = score_criterion(score_logits, score_y)
        loss_concede = concede_criterion(concede_logits, concede_y)
        loss = 0.5 * (loss_score + loss_concede)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    score_criterion: nn.Module,
    concede_criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows = []

    with torch.no_grad():
        for x_categ, x_cont, score_y, concede_y, match_ids, source_names, action_indices in tqdm(
            loader, desc=desc, leave=False
        ):
            x_categ = x_categ.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            score_y = score_y.to(device, non_blocking=True)
            concede_y = concede_y.to(device, non_blocking=True)

            score_logits, concede_logits = model(x_categ, x_cont)
            loss_score = score_criterion(score_logits, score_y)
            loss_concede = concede_criterion(concede_logits, concede_y)
            loss = 0.5 * (loss_score + loss_concede)
            total_loss += float(loss.item())

            score_prob = torch.sigmoid(score_logits)
            concede_prob = torch.sigmoid(concede_logits)

            for i in range(len(match_ids)):
                rows.append(
                    {
                        "match_id": int(match_ids[i]),
                        "source_name": str(source_names[i]),
                        "action_index": int(action_indices[i]),
                        "score_label": int(score_y[i].cpu().item()),
                        "concede_label": int(concede_y[i].cpu().item()),
                        "score_prob": float(score_prob[i].cpu().item()),
                        "concede_prob": float(concede_prob[i].cpu().item()),
                    }
                )

    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


def evaluate_task(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict[str, float]:
    preds = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "f1": binary_f1_score(y_true, preds.tolist()),
        "roc_auc": binary_roc_auc(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "positive_rate": float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
    }


def find_best_threshold(y_true: List[int], y_prob: List[float]) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.05):
        preds = (np.asarray(y_prob) >= threshold).astype(int)
        f1 = binary_f1_score(y_true, preds.tolist())
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def add_vaep_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    if len(pred_df) == 0:
        pred_df["next_score_prob"] = []
        pred_df["next_concede_prob"] = []
        pred_df["vaep_offense"] = []
        pred_df["vaep_defense"] = []
        pred_df["vaep_value"] = []
        return pred_df

    df = pred_df.sort_values(["match_id", "action_index"]).reset_index(drop=True).copy()
    same_match_next = df["match_id"].shift(-1) == df["match_id"]
    df["next_score_prob"] = np.where(same_match_next, df["score_prob"].shift(-1), np.nan)
    df["next_concede_prob"] = np.where(same_match_next, df["concede_prob"].shift(-1), np.nan)

    df["vaep_offense"] = df["next_score_prob"] - df["score_prob"]
    df["vaep_defense"] = df["concede_prob"] - df["next_concede_prob"]
    df["vaep_value"] = df["vaep_offense"] + df["vaep_defense"]
    df[["vaep_offense", "vaep_defense", "vaep_value"]] = df[["vaep_offense", "vaep_defense", "vaep_value"]].fillna(0.0)
    return df


def summarize_split(split_name: str, df: pd.DataFrame, score_threshold: float, concede_threshold: float) -> List[Dict[str, object]]:
    score_metrics = evaluate_task(df["score_label"].astype(int).tolist(), df["score_prob"].astype(float).tolist(), score_threshold)
    concede_metrics = evaluate_task(
        df["concede_label"].astype(int).tolist(),
        df["concede_prob"].astype(float).tolist(),
        concede_threshold,
    )

    rows = []
    for task, threshold, metrics in [
        ("score", score_threshold, score_metrics),
        ("concede", concede_threshold, concede_metrics),
    ]:
        rows.append(
            {
                "split": split_name,
                "task": task,
                "threshold": float(threshold),
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "positive_rate": metrics["positive_rate"],
            }
        )
    return rows


def choose_prediction_output_path(base_path: str, output_format: str) -> str:
    if output_format == "csv_gz":
        return base_path if base_path.endswith(".gz") else f"{base_path}.gz"
    if output_format == "parquet":
        stem, _ = os.path.splitext(base_path)
        return f"{stem}.parquet"
    return base_path


def save_predictions_compact(pred_df: pd.DataFrame, output_path: str, output_format: str) -> None:
    if output_format == "csv_gz":
        pred_df.to_csv(output_path, index=False, encoding="utf-8", compression="gzip")
        return
    if output_format == "parquet":
        pred_df.to_parquet(output_path, index=False)
        return
    pred_df.to_csv(output_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAEP baseline (score/concede multitask MLP).")
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD_ACTIONS, help="Future action window size for labels.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use. auto prefers CUDA when available.",
    )
    parser.add_argument(
        "--max-train-matches",
        type=int,
        default=None,
        help="Optional cap on number of training match files for quick debugging.",
    )
    parser.add_argument(
        "--max-test-matches",
        type=int,
        default=None,
        help="Optional cap on number of test match files for quick debugging.",
    )
    parser.add_argument(
        "--prediction-splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to export predictions for. Options: train,val,test",
    )
    parser.add_argument(
        "--prediction-format",
        type=str,
        default="csv_gz",
        choices=["csv", "csv_gz", "parquet"],
        help="Prediction output format. csv_gz is space-efficient and default.",
    )
    parser.add_argument(
        "--include-source-name",
        action="store_true",
        help="Include source_name column in exported predictions (increases file size).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation/export using an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=CHECKPOINT_OUTPUT,
        help="Checkpoint path used for eval-only, and as training save/load target.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available in this environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve checkpoint path from either explicit path or bare filename."""
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    fallback = os.path.join("checkpoints", os.path.basename(checkpoint_path))
    if os.path.exists(fallback):
        return fallback

    return checkpoint_path


def ensure_non_empty_split(
    train_actions: List[Dict[str, object]],
    val_actions: List[Dict[str, object]],
    seed: int = SEED,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if len(train_actions) > 0 and len(val_actions) > 0:
        return train_actions, val_actions

    merged = list(train_actions) + list(val_actions)
    if len(merged) <= 1:
        return merged, merged

    rng = random.Random(seed)
    rng.shuffle(merged)
    cut = max(1, int(0.9 * len(merged)))
    cut = min(cut, len(merged) - 1)
    return merged[:cut], merged[cut:]


def main() -> None:
    global DEVICE
    args = parse_args()
    DEVICE = resolve_device(args.device)
    set_seed(SEED)
    args.checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    use_cache = not bool(args.max_train_matches)
    print(f"DEVICE = {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"LOOKAHEAD = {args.lookahead}")
    print(f"CHECKPOINT_PATH = {args.checkpoint_path}")

    selected_splits = [s.strip() for s in args.prediction_splits.split(",") if s.strip()]
    valid_splits = [s for s in selected_splits if s in {"train", "val", "test"}]
    if not valid_splits:
        valid_splits = ["val", "test"]

    # ---- eval-only fast path: load metadata from checkpoint, skip train preprocessing ----
    if args.eval_only:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for eval-only: {args.checkpoint_path}")
        print(f"Eval-only mode: loading checkpoint {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
        cat_cols = checkpoint["cat_cols"]
        cont_cols = checkpoint["cont_cols"]
        cat_maps = checkpoint["cat_maps"]
        cat_dims = checkpoint["cat_dims"]
        cont_mean_std = checkpoint["cont_mean_std"]

        need_train_data = "train" in valid_splits
        need_test_data = "test" in valid_splits
        need_val_data = "val" in valid_splits

        use_pin_memory = DEVICE.type == "cuda"

        # Helper: make empty placeholder loader
        def _empty_loader() -> DataLoader:
            return DataLoader(
                VAEPTabularDataset([], [], cat_cols, cont_cols, cat_maps),
                batch_size=args.batch_size,
                shuffle=False,
            )

        if need_train_data or need_val_data:
            train_actions, val_actions = load_or_build_split_actions(
                TRAIN_ROOT,
                lookahead_k=args.lookahead,
                val_ratio=VAL_RATIO,
                seed=SEED,
                max_match_files=args.max_train_matches if args.max_train_matches is not None else MAX_TRAIN_MATCH_FILES,
                use_cache=use_cache,
            )
            if not need_train_data:
                del train_actions
                train_actions = []
        else:
            train_actions, val_actions = [], []

        if need_test_data:
            test_actions = load_or_build_actions(
                TEST_ROOT,
                lookahead_k=args.lookahead,
                max_match_files=args.max_test_matches if args.max_test_matches is not None else MAX_TEST_MATCH_FILES,
                use_cache=use_cache,
            )
        else:
            test_actions = []

        print(
            f"train actions: {len(train_actions)} | val actions: {len(val_actions)} | "
            f"test actions: {len(test_actions)}"
        )

        score_criterion = nn.BCEWithLogitsLoss()
        concede_criterion = nn.BCEWithLogitsLoss()

        model = VAEPMultitaskMLP(cat_dims, len(cont_cols), cont_mean_std).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def _make_loader(actions: list) -> DataLoader:
            indices = list(range(len(actions)))
            ds = VAEPTabularDataset(indices, actions, cat_cols, cont_cols, cat_maps)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_pin_memory)

        train_loader = _make_loader(train_actions) if need_train_data else _empty_loader()
        val_loader = _make_loader(val_actions) if need_val_data else _empty_loader()
        test_loader = _make_loader(test_actions) if need_test_data else _empty_loader()

        if need_train_data:
            _, train_pred = predict_dataset(model, train_loader, score_criterion, concede_criterion,
                                            device=DEVICE, desc="Predict train")
        else:
            train_pred = pd.DataFrame(columns=["match_id","source_name","action_index",
                                                "score_label","concede_label","score_prob","concede_prob"])

        if need_val_data:
            _, val_pred = predict_dataset(model, val_loader, score_criterion, concede_criterion,
                                          device=DEVICE, desc="Predict val")
        else:
            val_pred = pd.DataFrame(columns=["match_id","source_name","action_index",
                                             "score_label","concede_label","score_prob","concede_prob"])

        if need_test_data:
            _, test_pred = predict_dataset(model, test_loader, score_criterion, concede_criterion,
                                           device=DEVICE, desc="Predict test")
        else:
            test_pred = pd.DataFrame(columns=["match_id","source_name","action_index",
                                              "score_label","concede_label","score_prob","concede_prob"])

        _export_predictions_and_metrics(
            args, valid_splits, train_pred, val_pred, test_pred
        )
        return

    # ---- training path ----
    train_actions, val_actions = load_or_build_split_actions(
        TRAIN_ROOT,
        lookahead_k=args.lookahead,
        val_ratio=VAL_RATIO,
        seed=SEED,
        max_match_files=args.max_train_matches if args.max_train_matches is not None else MAX_TRAIN_MATCH_FILES,
        use_cache=use_cache,
    )
    test_actions = load_or_build_actions(
        TEST_ROOT,
        lookahead_k=args.lookahead,
        max_match_files=args.max_test_matches if args.max_test_matches is not None else MAX_TEST_MATCH_FILES,
        use_cache=use_cache,
    )
    print(
        f"train actions: {len(train_actions)} | val actions: {len(val_actions)} | "
        f"test actions: {len(test_actions)}"
    )

    cat_cols = list(DEFAULT_CAT_COLS)
    cont_cols = list(DEFAULT_CONT_COLS)
    cat_maps, cat_dims = build_cat_maps(train_actions, cat_cols)
    cont_mean_std = compute_cont_stats(train_actions, cont_cols)

    all_actions_combined = train_actions + val_actions + test_actions
    train_indices = list(range(len(train_actions)))
    val_indices = list(range(len(train_actions), len(train_actions) + len(val_actions)))
    test_indices = list(range(len(train_actions) + len(val_actions), len(all_actions_combined)))
    
    train_dataset = VAEPTabularDataset(train_indices, all_actions_combined, cat_cols, cont_cols, cat_maps)
    val_dataset = VAEPTabularDataset(val_indices, all_actions_combined, cat_cols, cont_cols, cat_maps)
    test_dataset = VAEPTabularDataset(test_indices, all_actions_combined, cat_cols, cont_cols, cat_maps)

    use_pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    model = VAEPMultitaskMLP(cat_dims, len(cont_cols), cont_mean_std).to(DEVICE)

    score_pos_weight = compute_pos_weight([int(a["score_label"]) for a in train_actions])
    concede_pos_weight = compute_pos_weight([int(a["concede_label"]) for a in train_actions])
    print(f"score_pos_weight = {score_pos_weight:.4f} | concede_pos_weight = {concede_pos_weight:.4f}")

    score_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(score_pos_weight, dtype=torch.float32, device=DEVICE))
    concede_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(concede_pos_weight, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            score_criterion,
            concede_criterion,
            device=DEVICE,
        )
        val_loss, val_df = predict_dataset(
            model,
            val_loader,
            score_criterion,
            concede_criterion,
            device=DEVICE,
            desc="Val batches",
        )

        score_logloss = binary_log_loss(val_df["score_label"].astype(int).tolist(), val_df["score_prob"].astype(float).tolist())
        concede_logloss = binary_log_loss(
            val_df["concede_label"].astype(int).tolist(),
            val_df["concede_prob"].astype(float).tolist(),
        )
        objective = 0.5 * (score_logloss + concede_logloss)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_score_logloss={score_logloss:.4f} | val_concede_logloss={concede_logloss:.4f}"
        )
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_score_ll=f"{score_logloss:.4f}",
            val_concede_ll=f"{concede_logloss:.4f}",
        )

        if objective < best_val:
            best_val = objective
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "cat_maps": cat_maps,
                    "cat_dims": cat_dims,
                    "cont_mean_std": cont_mean_std,
                    "lookahead": int(args.lookahead),
                    "epoch": int(epoch),
                },
                args.checkpoint_path,
            )
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "train" in valid_splits:
        _, train_pred = predict_dataset(
            model,
            train_loader,
            score_criterion,
            concede_criterion,
            device=DEVICE,
            desc="Predict train",
        )
    else:
        train_pred = pd.DataFrame(columns=[
            "match_id",
            "source_name",
            "action_index",
            "score_label",
            "concede_label",
            "score_prob",
            "concede_prob",
        ])
    _, val_pred = predict_dataset(
        model,
        val_loader,
        score_criterion,
        concede_criterion,
        device=DEVICE,
        desc="Predict val",
    )
    _, test_pred = predict_dataset(
        model,
        test_loader,
        score_criterion,
        concede_criterion,
        device=DEVICE,
        desc="Predict test",
    )

    _export_predictions_and_metrics(args, valid_splits, train_pred, val_pred, test_pred)


def _export_predictions_and_metrics(
    args,
    valid_splits: List[str],
    train_pred: "pd.DataFrame",
    val_pred: "pd.DataFrame",
    test_pred: "pd.DataFrame",
) -> None:
    # Determine thresholds from val (fall back to 0.5 if val is empty)
    if len(val_pred) > 0:
        score_thr, _ = find_best_threshold(
            val_pred["score_label"].astype(int).tolist(), val_pred["score_prob"].astype(float).tolist()
        )
        concede_thr, _ = find_best_threshold(
            val_pred["concede_label"].astype(int).tolist(), val_pred["concede_prob"].astype(float).tolist()
        )
    else:
        score_thr, concede_thr = 0.5, 0.5

    pred_by_split = {"train": train_pred, "val": val_pred, "test": test_pred}
    for split_name, df_split in pred_by_split.items():
        df_split["split"] = split_name

    all_pred = pd.concat([pred_by_split[s] for s in valid_splits], ignore_index=True)
    all_pred = add_vaep_columns(all_pred)

    if not args.include_source_name and "source_name" in all_pred.columns:
        all_pred = all_pred.drop(columns=["source_name"])

    output_path = choose_prediction_output_path(PREDICTIONS_OUTPUT, args.prediction_format)
    free_bytes = shutil.disk_usage(".").free
    if free_bytes < 2 * 1024**3:
        print(
            f"Warning: low disk space ({free_bytes / 1024**3:.2f} GB free). "
            f"Exporting selected splits only: {','.join(valid_splits)}"
        )
    save_predictions_compact(all_pred, output_path, args.prediction_format)

    metrics_rows: List[Dict[str, object]] = []
    if len(train_pred) > 0:
        metrics_rows.extend(summarize_split("train", train_pred, score_thr, concede_thr))
    if len(val_pred) > 0:
        metrics_rows.extend(summarize_split("val", val_pred, score_thr, concede_thr))
    if len(test_pred) > 0:
        metrics_rows.extend(summarize_split("test", test_pred, score_thr, concede_thr))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"Saved predictions: {output_path}")
    print(f"Saved metrics: {METRICS_OUTPUT}")
    print(f"score_threshold={score_thr:.2f} | concede_threshold={concede_thr:.2f}")


if __name__ == "__main__":
    main()



