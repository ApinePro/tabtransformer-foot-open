import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange, repeat
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
    except Exception:
        pass


# =========================
# 配置区
# =========================
def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def with_exp_suffix(filename: str) -> str:
    exp_id = os.getenv("EXPERIMENT_ID", "").strip()
    if not exp_id:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{exp_id}{ext}"


TRAIN_ROOT = "dataset"
TEST_ROOT = "dataset-test"

def _resolve_pretrain_checkpoint() -> str:
    pretrain_exp_id = os.getenv("PRETRAIN_EXPERIMENT_ID", None)
    if pretrain_exp_id is not None:
        # 显式指定：空字符串 = 用无 suffix 的基础 checkpoint
        if pretrain_exp_id.strip():
            return os.path.join("checkpoints", f"tabtransformer_row_mask_pretrain_{pretrain_exp_id.strip()}.pt")
        else:
            return os.path.join("checkpoints", "tabtransformer_row_mask_pretrain.pt")
    return os.path.join("checkpoints", with_exp_suffix("tabtransformer_row_mask_pretrain.pt"))

PRETRAIN_CHECKPOINT = _resolve_pretrain_checkpoint()
XG_CHECKPOINT = os.path.join("checkpoints", with_exp_suffix("xg_tabtransformer_best.pt"))
METRICS_OUTPUT = with_exp_suffix("xg_tabtransformer_metrics.csv")
PREDICTIONS_OUTPUT = with_exp_suffix("xg_tabtransformer_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MAX_TRAIN_MATCH_FILES: Optional[int] = None
MAX_TEST_MATCH_FILES: Optional[int] = None
VAL_RATIO = 0.1
MAX_SEQ_LEN = 48
BATCH_SIZE = 8
EPOCHS = int(os.getenv("XG_EPOCHS", "5"))
LR = 1e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
NUM_SEQUENCE_LAYERS = 2
NUM_SEQUENCE_HEADS = 4
DROPOUT = 0.1

USE_PRETRAINED_ENCODER = env_flag("XG_USE_PRETRAINED_ENCODER", True)
FREEZE_EVENT_ENCODER = env_flag("XG_FREEZE_EVENT_ENCODER", False)
USE_CLASS_WEIGHTS = True
MAX_POS_WEIGHT = 20.0
WEIGHT_MODE = "sqrt_capped"  # 可选: ratio / sqrt_capped / capped

# info policy for experiment matrix
# - XG_ALLOW_SHOT_END_FOR_SHOT=1: allow using shot_end_location for shot rows (full-info)
# - XG_REMOVE_SHOT_DURATION=1: zero-out duration for shot rows
XG_ALLOW_SHOT_END_FOR_SHOT = env_flag("XG_ALLOW_SHOT_END_FOR_SHOT", False)
XG_REMOVE_SHOT_DURATION = env_flag("XG_REMOVE_SHOT_DURATION", False)

# 与预训练保持一致，默认去掉 player / team identity，减少“球员强弱”偏置。
# 但保留 role-based 特征来区分序列中的两支球队。
USE_PLAYER_IDENTITY = False
USE_TEAM_IDENTITY = False

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_X_BINS = 16
GRID_Y_BINS = 12
GOAL_CENTER_Y = 40.0
OPEN_PLAY_PATTERNS = {"Regular Play", "From Counter"}
STOPPAGE_TYPES = {"Half End", "Half Start", "Substitution", "Injury Stoppage", "Tactical Shift"}
SKIP_TYPES = {"Starting XI", "Half Start"}

DEFAULT_CAT_COLS = [
    "type",
    "play_pattern",
    "position",
    "team_role",
    "sequence_team_role",
    "start_zone",
    "end_zone",
    "pass_height",
    "pass_type",
    "pass_body_part",
    "pass_technique",
    "pass_outcome",
    "pass_cross",
    "pass_cut_back",
    "pass_switch",
    "pass_through_ball",
    "dribble_outcome",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "foul_committed_type",
    "goalkeeper_type",
    "goalkeeper_outcome",
    "counterpress",
    "shot_type",
    "shot_body_part",
    "shot_technique",
    "shot_first_time",
    "shot_one_on_one",
]
if USE_TEAM_IDENTITY:
    DEFAULT_CAT_COLS.extend(["team", "possession_team"])
if USE_PLAYER_IDENTITY:
    DEFAULT_CAT_COLS.append("player")

DEFAULT_CONT_COLS = [
    "period",
    "minute",
    "second",
    "duration",
    "location_x",
    "location_y",
    "end_x",
    "end_y",
    "delta_x",
    "delta_y",
    "distance_to_goal",
    "angle_to_goal",
    "pass_length",
    "pass_angle",
    "under_pressure_num",
]

RAW_EVENT_COLUMNS = [
    "match_id",
    "index",
    "type",
    "team",
    "possession_team",
    "player",
    "position",
    "play_pattern",
    "period",
    "minute",
    "second",
    "duration",
    "location",
    "pass_end_location",
    "carry_end_location",
    "shot_end_location",
    "pass_height",
    "pass_type",
    "pass_body_part",
    "pass_technique",
    "pass_outcome",
    "pass_cross",
    "pass_cut_back",
    "pass_switch",
    "pass_through_ball",
    "pass_angle",
    "pass_length",
    "shot_type",
    "shot_body_part",
    "shot_technique",
    "shot_first_time",
    "shot_one_on_one",
    "shot_outcome",
    "dribble_outcome",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "foul_committed_type",
    "goalkeeper_type",
    "goalkeeper_outcome",
    "counterpress",
    "under_pressure",
    "out",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    return str(value)


def safe_num(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_xy(value) -> Tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def discretize_zone(x: float, y: float) -> str:
    x_idx = min(GRID_X_BINS - 1, max(0, int(np.floor(x / (PITCH_LENGTH / GRID_X_BINS)))))
    y_idx = min(GRID_Y_BINS - 1, max(0, int(np.floor(y / (PITCH_WIDTH / GRID_Y_BINS)))))
    return f"zone_{x_idx:02d}_{y_idx:02d}"


def distance_and_angle_to_goal(x: float, y: float) -> Tuple[float, float]:
    dx = max(PITCH_LENGTH - x, 1e-6)
    dy = abs(GOAL_CENTER_Y - y)
    distance = float(np.sqrt(dx**2 + dy**2))
    goal_width = 8.0
    angle = np.arctan2(goal_width * dx, max(dx**2 + dy**2 - (goal_width / 2.0) ** 2, 1e-6))
    return distance, float(max(angle, 0.0))


def is_open_play_event(row: pd.Series) -> bool:
    return safe_text(row.get("play_pattern")) in OPEN_PLAY_PATTERNS


def is_ball_out_event(row: pd.Series) -> bool:
    if bool(row.get("out", False)):
        return True
    etype = safe_text(row.get("type"))
    if etype in STOPPAGE_TYPES:
        return True
    if safe_text(row.get("pass_outcome")) == "Out":
        return True
    if safe_text(row.get("interception_outcome")) == "Lost Out":
        return True
    return False


def is_definitive_turnover(row: pd.Series) -> bool:
    etype = safe_text(row.get("type"))
    interception_outcome = safe_text(row.get("interception_outcome"))
    if etype == "Interception" and interception_outcome in {"Won", "Success In Play"}:
        return True
    if etype == "Clearance":
        return True
    return False


def build_event_features(
    row: pd.Series,
    cat_cols: List[str],
    cont_cols: List[str],
    sequence_team: Optional[str] = None,
) -> Dict[str, object]:
    etype = safe_text(row.get("type"))
    location_x, location_y = safe_xy(row.get("location"))

    # For strict pre-shot modeling, avoid shot_end_location on shot events unless full-info is enabled.
    end_x, end_y = location_x, location_y
    end_cols = ["pass_end_location", "carry_end_location"]
    if etype != "Shot" or XG_ALLOW_SHOT_END_FOR_SHOT:
        end_cols.append("shot_end_location")

    for end_col in end_cols:
        val = row.get(end_col)
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
            end_x, end_y = safe_xy(val)
            break

    distance_to_goal, angle_to_goal = distance_and_angle_to_goal(location_x, location_y)
    team_text = safe_text(row.get("team"))
    possession_text = safe_text(row.get("possession_team"))
    team_role = "in_possession" if team_text == possession_text else "out_of_possession"
    if sequence_team is None or safe_text(sequence_team) == "__MISSING__":
        sequence_team_role = "__MISSING__"
    else:
        sequence_team_role = "attacking_team" if team_text == safe_text(sequence_team) else "opponent_team"

    all_values = {
        "type": etype,
        "play_pattern": safe_text(row.get("play_pattern")),
        "position": safe_text(row.get("position")),
        "team_role": team_role,
        "sequence_team_role": sequence_team_role,
        "start_zone": discretize_zone(location_x, location_y),
        "end_zone": discretize_zone(end_x, end_y),
        "pass_height": safe_text(row.get("pass_height")),
        "pass_type": safe_text(row.get("pass_type")),
        "pass_body_part": safe_text(row.get("pass_body_part")),
        "pass_technique": safe_text(row.get("pass_technique")),
        "pass_outcome": safe_text(row.get("pass_outcome")),
        "pass_cross": safe_text(row.get("pass_cross")),
        "pass_cut_back": safe_text(row.get("pass_cut_back")),
        "pass_switch": safe_text(row.get("pass_switch")),
        "pass_through_ball": safe_text(row.get("pass_through_ball")),
        "shot_type": safe_text(row.get("shot_type")),
        "shot_body_part": safe_text(row.get("shot_body_part")),
        "shot_technique": safe_text(row.get("shot_technique")),
        "shot_first_time": safe_text(row.get("shot_first_time")),
        "shot_one_on_one": safe_text(row.get("shot_one_on_one")),
        "dribble_outcome": safe_text(row.get("dribble_outcome")),
        "duel_type": safe_text(row.get("duel_type")),
        "duel_outcome": safe_text(row.get("duel_outcome")),
        "interception_outcome": safe_text(row.get("interception_outcome")),
        "foul_committed_type": safe_text(row.get("foul_committed_type")),
        "goalkeeper_type": safe_text(row.get("goalkeeper_type")),
        "goalkeeper_outcome": safe_text(row.get("goalkeeper_outcome")),
        "counterpress": safe_text(row.get("counterpress")),
        "team": safe_text(row.get("team")),
        "possession_team": safe_text(row.get("possession_team")),
        "player": safe_text(row.get("player")),
        "period": safe_num(row.get("period")),
        "minute": safe_num(row.get("minute")),
        "second": safe_num(row.get("second")),
        "duration": 0.0 if (XG_REMOVE_SHOT_DURATION and etype == "Shot") else safe_num(row.get("duration")),
        "location_x": location_x,
        "location_y": location_y,
        "end_x": end_x,
        "end_y": end_y,
        "delta_x": end_x - location_x,
        "delta_y": end_y - location_y,
        "distance_to_goal": distance_to_goal,
        "angle_to_goal": angle_to_goal,
        "pass_length": safe_num(row.get("pass_length")),
        "pass_angle": safe_num(row.get("pass_angle")),
        "under_pressure_num": float(bool(row.get("under_pressure", False))),
    }
    return {col: all_values[col] for col in cat_cols + cont_cols}


def load_sequences_from_root(
    root_dir: str,
    cat_cols: List[str],
    cont_cols: List[str],
    max_match_files: Optional[int] = None,
) -> List[Dict[str, object]]:
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

    all_sequences: List[Dict[str, object]] = []
    for path, source_name in tqdm(event_files, desc=f"Segmenting {root_dir}"):
        try:
            df = pd.read_parquet(path, columns=RAW_EVENT_COLUMNS)
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in RAW_EVENT_COLUMNS if c in df.columns]
            df = df[keep].copy()

        # Pre-clean shot endpoint unless full-info mode explicitly allows it.
        if (not XG_ALLOW_SHOT_END_FOR_SHOT) and "type" in df.columns and "shot_end_location" in df.columns:
            shot_mask = df["type"].astype(str).eq("Shot")
            if bool(shot_mask.any()):
                df = df.copy()
                df.loc[shot_mask, "shot_end_location"] = None

        all_sequences.extend(segment_match_into_shot_sequences(df, cat_cols, cont_cols, source_name))
    return all_sequences


def segment_match_into_shot_sequences(
    match_events: pd.DataFrame,
    cat_cols: List[str],
    cont_cols: List[str],
    source_name: str,
) -> List[Dict[str, object]]:
    if len(match_events) == 0:
        return []

    df = match_events.sort_values(["period", "minute", "second", "index"]).reset_index(drop=True)
    match_id = int(df["match_id"].iloc[0]) if "match_id" in df.columns else -1

    sequences: List[Dict[str, object]] = []
    current_events: List[Dict[str, object]] = []
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
            current_events = []
            current_team = None
            opponent_touch_streak = 0
        last_period = period

        # 只保留 open-play / counter 的进攻过程。遇到定位球等则重置。
        if not is_open_play_event(row):
            current_events = []
            current_team = None
            opponent_touch_streak = 0
            continue

        if not current_events:
            current_team = team
            opponent_touch_streak = 0
        else:
            if team != "__MISSING__" and current_team not in {None, "__MISSING__"}:
                if team == current_team:
                    opponent_touch_streak = 0
                else:
                    opponent_touch_streak += 1

        sequence_team = current_team if current_team not in {None, '__MISSING__'} else team
        current_events.append(build_event_features(row, cat_cols, cont_cols, sequence_team=sequence_team))

        if etype == "Shot":
            label = 1 if safe_text(row.get("shot_outcome")) == "Goal" else 0
            sequences.append(
                {
                    "match_id": match_id,
                    "sequence_id": shot_idx,
                    "source_name": source_name,
                    "label": label,
                    "shot_outcome": safe_text(row.get("shot_outcome")),
                    "sequence_len": len(current_events),
                    "events": current_events[-MAX_SEQ_LEN :].copy(),
                }
            )
            shot_idx += 1
            current_events = []
            current_team = None
            opponent_touch_streak = 0
            continue

        if is_ball_out_event(row) or is_definitive_turnover(row) or opponent_touch_streak >= 3:
            current_events = []
            current_team = None
            opponent_touch_streak = 0

    return sequences


def split_train_val_by_match(
    sequences: List[Dict[str, object]], val_ratio: float = VAL_RATIO, seed: int = SEED
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    match_ids = sorted({int(seq["match_id"]) for seq in sequences})
    rng = random.Random(seed)
    rng.shuffle(match_ids)

    n_val = max(1, int(len(match_ids) * val_ratio))
    val_ids = set(match_ids[:n_val])
    train = [seq for seq in sequences if int(seq["match_id"]) not in val_ids]
    val = [seq for seq in sequences if int(seq["match_id"]) in val_ids]
    return train, val


def build_cat_maps_from_sequences(
    sequences: List[Dict[str, object]], cat_cols: List[str]
) -> Tuple[Dict[str, Dict[str, int]], List[int]]:
    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        uniques = sorted({safe_text(event.get(col)) for seq in sequences for event in seq["events"]})
        cat_maps[col] = {value: idx + 1 for idx, value in enumerate(uniques)}
    cat_dims = [len(cat_maps[col]) + 1 for col in cat_cols]
    return cat_maps, cat_dims


def compute_cont_stats(sequences: List[Dict[str, object]], cont_cols: List[str]) -> np.ndarray:
    stats = []
    for col in cont_cols:
        values = [safe_num(event.get(col)) for seq in sequences for event in seq["events"]]
        arr = np.asarray(values, dtype=np.float32)
        mean = float(arr.mean()) if len(arr) > 0 else 0.0
        std = float(arr.std()) if len(arr) > 1 else 1.0
        if std < 1e-6:
            std = 1.0
        stats.append((mean, std))
    return np.asarray(stats, dtype=np.float32)


class ShotSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[Dict[str, object]],
        cat_cols: List[str],
        cont_cols: List[str],
        cat_maps: Dict[str, Dict[str, int]],
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.sequences = sequences
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.cat_maps = cat_maps
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sample = self.sequences[idx]
        events = list(sample["events"])[-self.max_seq_len :]
        label = int(sample["label"])

        x_categ = np.zeros((self.max_seq_len, len(self.cat_cols)), dtype=np.int64)
        x_cont = np.zeros((self.max_seq_len, len(self.cont_cols)), dtype=np.float32)
        valid_mask = np.zeros(self.max_seq_len, dtype=np.bool_)

        for i, event in enumerate(events):
            cat_values = [self.cat_maps[col].get(safe_text(event.get(col)), 0) for col in self.cat_cols]
            cont_values = [safe_num(event.get(col)) for col in self.cont_cols]
            x_categ[i] = np.asarray(cat_values, dtype=np.int64)
            x_cont[i] = np.asarray(cont_values, dtype=np.float32)
            valid_mask[i] = True

        return (
            torch.tensor(x_categ, dtype=torch.long),
            torch.tensor(x_cont, dtype=torch.float32),
            torch.tensor(valid_mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.long),
            sample["match_id"],
            sample["source_name"],
            sample["sequence_id"],
        )


class TabEventEncoder(nn.Module):
    def __init__(self, cat_dims: List[int], num_cont: int, cont_mean_std: np.ndarray):
        super().__init__()
        self.backbone = TabTransformer(
            categories=tuple(cat_dims),
            num_continuous=num_cont,
            dim=32,
            dim_out=sum(cat_dims),
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
            continuous_mean_std=torch.tensor(cont_mean_std, dtype=torch.float32),
        )
        self.model_dim = 32
        self.output_dim = len(cat_dims) * self.model_dim + num_cont

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        xs = []

        if self.backbone.num_unique_categories > 0:
            is_special_token = x_categ < 0
            categ_embed = self.backbone.categorical_embeds(x_categ.clamp_min(0), sum_discrete_sets=False)

            if is_special_token.any():
                special_token_ids = (x_categ + 1).abs().clamp_max(self.backbone.num_special_tokens - 1)
                special_embed = self.backbone.special_token_embed(special_token_ids)
                categ_embed = torch.where(is_special_token[..., None], special_embed, categ_embed)

            if self.backbone.use_shared_categ_embed:
                shared_categ_embed = repeat(
                    self.backbone.shared_category_embed, "n d -> b n d", b=categ_embed.shape[0]
                )
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)

            x, _ = self.backbone.transformer(categ_embed, return_attn=True)
            flat_categ = rearrange(x, "b n d -> b (n d)")
            xs.append(flat_categ)

        if self.backbone.num_continuous > 0:
            if self.backbone.continuous_mean_std is not None:
                mean, std = self.backbone.continuous_mean_std.to(x_cont.device).unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.backbone.norm(x_cont)
            xs.append(normed_cont)

        return torch.cat(xs, dim=-1)


class SequenceXGModel(nn.Module):
    def __init__(self, cat_dims: List[int], num_cont: int, cont_mean_std: np.ndarray):
        super().__init__()
        self.event_encoder = TabEventEncoder(cat_dims, num_cont, cont_mean_std)
        self.event_proj = nn.Sequential(
            nn.Linear(self.event_encoder.output_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_SEQ_LEN + 1, HIDDEN_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_SEQUENCE_HEADS,
            dim_feedforward=HIDDEN_DIM * 4,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_SEQUENCE_LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 2))

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_cat = x_categ.shape
        _, _, num_cont = x_cont.shape

        flat_categ = x_categ.view(batch_size * seq_len, num_cat)
        flat_cont = x_cont.view(batch_size * seq_len, num_cont)
        event_features = self.event_encoder(flat_categ, flat_cont)
        event_features = event_features.view(batch_size, seq_len, -1)
        event_tokens = self.event_proj(event_features)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, event_tokens], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        full_mask = torch.cat([cls_mask, valid_mask], dim=1)
        key_padding_mask = ~full_mask

        x = self.sequence_encoder(x, src_key_padding_mask=key_padding_mask)
        cls_repr = x[:, 0]
        return self.head(cls_repr)


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    if n_pos == 0 or not USE_CLASS_WEIGHTS:
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE)

    ratio = n_neg / max(n_pos, 1)
    if WEIGHT_MODE == "ratio":
        pos_weight = ratio
    elif WEIGHT_MODE == "capped":
        pos_weight = min(MAX_POS_WEIGHT, ratio)
    else:
        pos_weight = min(MAX_POS_WEIGHT, math.sqrt(ratio))

    return torch.tensor([1.0, float(pos_weight)], dtype=torch.float32, device=DEVICE)


def binary_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def binary_roc_auc(y_true: List[int], y_score: List[float]) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[pos].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_log_loss(y_true: List[int], y_prob: List[float], eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float32), eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss.mean())


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


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


def predict_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    rows = []

    with torch.no_grad():
        for x_categ, x_cont, valid_mask, labels, match_ids, source_names, sequence_ids in loader:
            x_categ = x_categ.to(DEVICE)
            x_cont = x_cont.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(x_categ, x_cont, valid_mask)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            total_loss += float(loss.item())
            for i in range(len(labels)):
                rows.append(
                    {
                        "match_id": int(match_ids[i]),
                        "source_name": str(source_names[i]),
                        "sequence_id": int(sequence_ids[i]),
                        "label": int(labels[i].cpu().item()),
                        "goal_prob": float(probs[i].cpu().item()),
                    }
                )

    return total_loss / max(len(loader), 1), pd.DataFrame(rows)


def evaluate_prediction_df(pred_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    preds = (pred_df["goal_prob"].values >= threshold).astype(int)
    labels = pred_df["label"].astype(int).tolist()
    probs = pred_df["goal_prob"].astype(float).tolist()
    return {
        "threshold": threshold,
        "f1": binary_f1_score(labels, preds.tolist()),
        "roc_auc": binary_roc_auc(labels, probs),
        "log_loss": binary_log_loss(labels, probs),
        "brier": brier_score(labels, probs),
        "positive_rate": float(pred_df["label"].mean()) if len(pred_df) > 0 else 0.0,
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0

    for x_categ, x_cont, valid_mask, labels, _, _, _ in tqdm(loader, desc="Train", leave=False):
        x_categ = x_categ.to(DEVICE)
        x_cont = x_cont.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(x_categ, x_cont, valid_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def maybe_load_pretrain_metadata(checkpoint_path: str = PRETRAIN_CHECKPOINT):
    if not USE_PRETRAINED_ENCODER or not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def maybe_load_pretrained_weights(model: SequenceXGModel, checkpoint_path: str = PRETRAIN_CHECKPOINT) -> None:
    if not USE_PRETRAINED_ENCODER or not os.path.exists(checkpoint_path):
        print("未找到预训练权重，将从头开始训练 xG 模型。")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.event_encoder.backbone.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"已加载预训练 event encoder: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    if FREEZE_EVENT_ENCODER:
        for param in model.event_encoder.parameters():
            param.requires_grad = False
        print("已冻结预训练 event encoder。")


def main():
    set_seed(SEED)
    print(f"DEVICE = {DEVICE}")
    print(
        f"XG_ALLOW_SHOT_END_FOR_SHOT={XG_ALLOW_SHOT_END_FOR_SHOT}, "
        f"XG_REMOVE_SHOT_DURATION={XG_REMOVE_SHOT_DURATION}"
    )
    print(f"PRETRAIN_CHECKPOINT={PRETRAIN_CHECKPOINT}")
    print(f"XG_CHECKPOINT={XG_CHECKPOINT}")

    pretrain_meta = maybe_load_pretrain_metadata()
    if pretrain_meta is not None:
        cat_cols = list(pretrain_meta["cat_cols"])
        cont_cols = list(pretrain_meta["cont_cols"])
        cat_maps = pretrain_meta["cat_maps"]
        cat_dims = list(pretrain_meta["cat_dims"])
        cont_mean_std = np.asarray(pretrain_meta["cont_mean_std"], dtype=np.float32)
        print("使用预训练阶段保存的特征定义与编码映射。")
    else:
        cat_cols = list(DEFAULT_CAT_COLS)
        cont_cols = list(DEFAULT_CONT_COLS)
        cat_maps = None
        cat_dims = None
        cont_mean_std = None
        print("未找到预训练元数据，将使用默认 xG 特征定义。")

    train_sequences_all = load_sequences_from_root(
        TRAIN_ROOT, cat_cols, cont_cols, max_match_files=MAX_TRAIN_MATCH_FILES
    )
    test_sequences = load_sequences_from_root(
        TEST_ROOT, cat_cols, cont_cols, max_match_files=MAX_TEST_MATCH_FILES
    )

    train_sequences, val_sequences = split_train_val_by_match(train_sequences_all, val_ratio=VAL_RATIO, seed=SEED)

    if cat_maps is None or cat_dims is None:
        cat_maps, cat_dims = build_cat_maps_from_sequences(train_sequences, cat_cols)
        cont_mean_std = compute_cont_stats(train_sequences, cont_cols)

    print(f"训练 shots: {len(train_sequences)} | 验证 shots: {len(val_sequences)} | 测试 shots: {len(test_sequences)}")
    train_labels = [int(seq["label"]) for seq in train_sequences]
    print(f"训练集进球率 = {np.mean(train_labels):.4f}")

    train_dataset = ShotSequenceDataset(train_sequences, cat_cols, cont_cols, cat_maps)
    val_dataset = ShotSequenceDataset(val_sequences, cat_cols, cont_cols, cat_maps)
    test_dataset = ShotSequenceDataset(test_sequences, cat_cols, cont_cols, cat_maps)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SequenceXGModel(cat_dims, len(cont_cols), cont_mean_std).to(DEVICE)
    maybe_load_pretrained_weights(model)

    class_weights = compute_class_weights(train_labels)
    print(f"Class weights = {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(os.path.dirname(XG_CHECKPOINT), exist_ok=True)
    best_val_auc = -1.0
    best_threshold = 0.5

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_pred_df = predict_model(model, val_loader, criterion)
        current_threshold, val_best_f1 = find_best_threshold(val_pred_df["label"].tolist(), val_pred_df["goal_prob"].tolist())
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
                },
                XG_CHECKPOINT,
            )
            print(f"已保存最佳 xG 模型: {XG_CHECKPOINT}")

    if os.path.exists(XG_CHECKPOINT):
        best_ckpt = torch.load(XG_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_threshold = float(best_ckpt.get("best_threshold", 0.5))

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

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    test_pred_df.to_csv(PREDICTIONS_OUTPUT, index=False)
    print(f"\n评估结果已保存到: {METRICS_OUTPUT}")
    print(f"每个 shot 的 xG 预测已保存到: {PREDICTIONS_OUTPUT}")
    print("goal_prob 列就是模型输出的 xG 近似值 P(goal | attack sequence ending in shot)。")


if __name__ == "__main__":
    main()
