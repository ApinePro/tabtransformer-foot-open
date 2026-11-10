import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


if hasattr(sys.stdout, "reconfigure"):
	try:
		sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
		sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
	except Exception:
		pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "dataset"
MAX_EVENT_FILES_FOR_TRAIN: Optional[int] = None
MAX_ROWS_FOR_TRAIN: int = 100000
MASK_PROB: float = 0.15
BATCH_SIZE: int = 256
PRETRAIN_EPOCHS: int = 20
SEED: int = 42

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_X_BINS = 16
GRID_Y_BINS = 12
GOAL_CENTER_Y = 40.0

RAW_EVENT_COLUMNS = [
	"type",
	"play_pattern",
	"position",
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
	"dribble_outcome",
	"duel_type",
	"duel_outcome",
	"interception_outcome",
	"foul_committed_type",
	"goalkeeper_type",
	"goalkeeper_outcome",
	"counterpress",
	"team",
	"possession_team",
	"player",
	"period",
	"minute",
	"second",
	"duration",
	"location",
	"pass_end_location",
	"carry_end_location",
	"shot_end_location",
	"under_pressure",
]

# 为了减少“球员强弱/身份”对事件表征的影响，默认不使用 player 身份特征。
USE_PLAYER_IDENTITY = False
# 不使用具体球队 identity，但保留 role-based 特征（如控球方 / 非控球方）来区分两队。
USE_TEAM_IDENTITY = False


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


# pretrain feature policies
# - PRETRAIN_STRICT_PRIOR=1: disable end_location-derived signals and duration.
# - PRETRAIN_FULL_INFO=1: keep full end_location and duration signals.
PRETRAIN_STRICT_PRIOR = env_flag("PRETRAIN_STRICT_PRIOR", False)
PRETRAIN_FULL_INFO = env_flag("PRETRAIN_FULL_INFO", False)

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, with_exp_suffix("tabtransformer_row_mask_pretrain.pt"))
LOSS_CURVE_OUTPUT = with_exp_suffix("loss_curve_row_mask_pretrain.png")


def safe_xy(value) -> Tuple[float, float]:
	if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
		return float(value[0]), float(value[1])
	return np.nan, np.nan


def safe_text(value) -> str:
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return "__MISSING__"
	return str(value)


def discretize_zone(x: float, y: float) -> str:
	if np.isnan(x) or np.isnan(y):
		return "__MISSING__"
	x_idx = min(GRID_X_BINS - 1, max(0, int(np.floor(x / (PITCH_LENGTH / GRID_X_BINS)))))
	y_idx = min(GRID_Y_BINS - 1, max(0, int(np.floor(y / (PITCH_WIDTH / GRID_Y_BINS)))))
	return f"zone_{x_idx:02d}_{y_idx:02d}"


def distance_and_angle_to_goal(x: float, y: float) -> Tuple[float, float]:
	if np.isnan(x) or np.isnan(y):
		return 0.0, 0.0
	dx = max(PITCH_LENGTH - x, 1e-6)
	dy = abs(GOAL_CENTER_Y - y)
	distance = float(np.sqrt(dx**2 + dy**2))
	goal_width = 8.0
	angle = np.arctan2(goal_width * dx, max(dx**2 + dy**2 - (goal_width / 2.0) ** 2, 1e-6))
	return distance, float(max(angle, 0.0))


def load_saved_events(
	save_root: str = DATASET_ROOT,
	limit_files: Optional[int] = None,
	max_rows: Optional[int] = None,
) -> pd.DataFrame:
	event_files = []
	for root, _, files in os.walk(save_root):
		if os.path.basename(root) == "events":
			for name in files:
				if name.endswith(".parquet"):
					event_files.append(os.path.join(root, name))

	event_files = sorted(event_files)
	if limit_files is not None:
		event_files = event_files[:limit_files]

	if not event_files:
		raise FileNotFoundError(
			f"未在 {save_root} 下找到已保存的 events parquet 文件，请先运行: python download_statsbomb_open_data.py"
		)

	# 打乱文件顺序，避免只读到排序靠前的联赛/比赛。
	rng = np.random.default_rng(SEED)
	event_files = list(rng.permutation(event_files))

	frames = []
	total_rows = 0
	for path in tqdm(event_files, desc="Loading saved events"):
		try:
			part = pd.read_parquet(path, columns=RAW_EVENT_COLUMNS)
		except Exception:
			part = pd.read_parquet(path)
			keep_cols = [c for c in RAW_EVENT_COLUMNS if c in part.columns]
			part = part[keep_cols].copy()

		if len(part) == 0:
			continue

		if max_rows is not None:
			remaining = max_rows - total_rows
			if remaining <= 0:
				break
			if len(part) > remaining:
				part = part.sample(n=remaining, random_state=SEED).reset_index(drop=True)

		frames.append(part)
		total_rows += len(part)

		if max_rows is not None and total_rows >= max_rows:
			break

	if not frames:
		raise RuntimeError("没有成功加载任何 events 数据")

	return pd.concat(frames, ignore_index=True)


def build_features(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
	df = df_raw.copy()

	if "location" in df.columns:
		loc_xy = df["location"].apply(safe_xy)
		df["location_x"] = loc_xy.apply(lambda t: t[0])
		df["location_y"] = loc_xy.apply(lambda t: t[1])
	else:
		df["location_x"] = np.nan
		df["location_y"] = np.nan

	if PRETRAIN_STRICT_PRIOR:
		# strict prior pretraining: do not use any end-location fields.
		end_col_candidates = []
	elif PRETRAIN_FULL_INFO:
		end_col_candidates = ["pass_end_location", "carry_end_location", "shot_end_location"]
	else:
		# task-prior pretraining: keep movement endpoints, but avoid shot end location.
		end_col_candidates = ["pass_end_location", "carry_end_location"]
	end_xy = pd.Series([(np.nan, np.nan)] * len(df), index=df.index)
	for col in end_col_candidates:
		if col in df.columns:
			has_value = end_xy.apply(lambda t: np.isnan(t[0])) & df[col].notna()
			end_xy.loc[has_value] = df.loc[has_value, col].apply(safe_xy)

	df["end_x"] = end_xy.apply(lambda t: t[0])
	df["end_y"] = end_xy.apply(lambda t: t[1])
	df["delta_x"] = (df["end_x"] - df["location_x"]).fillna(0.0)
	df["delta_y"] = (df["end_y"] - df["location_y"]).fillna(0.0)
	df["start_zone"] = [discretize_zone(x, y) for x, y in zip(df["location_x"], df["location_y"])]
	df["end_zone"] = [discretize_zone(x, y) for x, y in zip(df["end_x"], df["end_y"])]

	goal_geometry = [distance_and_angle_to_goal(x, y) for x, y in zip(df["location_x"], df["location_y"])]
	df["distance_to_goal"] = [g[0] for g in goal_geometry]
	df["angle_to_goal"] = [g[1] for g in goal_geometry]

	if "under_pressure" in df.columns:
		df["under_pressure_num"] = df["under_pressure"].fillna(False).astype(bool).astype(float)
	else:
		df["under_pressure_num"] = 0.0

	if "team" in df.columns and "possession_team" in df.columns:
		team_text = df["team"].apply(safe_text)
		possession_text = df["possession_team"].apply(safe_text)
		df["team_role"] = np.where(team_text == possession_text, "in_possession", "out_of_possession")
	else:
		df["team_role"] = "__MISSING__"

	# 行内 mask 重建：加入更多安全特征，但依然避免直接结果泄漏与 player/team identity。
	cat_cols = [
		"type",
		"play_pattern",
		"position",
		"team_role",
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
		cat_cols.extend(["team", "possession_team"])
	if USE_PLAYER_IDENTITY:
		cat_cols.append("player")

	cont_cols = [
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

	cat_cols = [c for c in cat_cols if c in df.columns]
	cont_cols = [c for c in cont_cols if c in df.columns]

	keep_cols = cat_cols + cont_cols
	df = df[keep_cols].copy()

	for col in cat_cols:
		df[col] = df[col].apply(safe_text)

	for col in cont_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

	if "duration" in df.columns:
		if PRETRAIN_STRICT_PRIOR:
			df["duration"] = 0.0
		elif not PRETRAIN_FULL_INFO and "type" in df.columns:
			shot_mask = df["type"].astype(str).eq("Shot")
			if bool(shot_mask.any()):
				df.loc[shot_mask, "duration"] = 0.0

	return df, cat_cols, cont_cols


def encode_categories(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], List[int]]:
	cat_maps: Dict[str, Dict[str, int]] = {}
	for col in cat_cols:
		uniques = df[col].astype(str).unique().tolist()
		cat_maps[col] = {v: i + 1 for i, v in enumerate(uniques)}
		df[col] = df[col].astype(str).map(cat_maps[col]).fillna(0).astype(int)

	cat_dims = [len(cat_maps[c]) + 1 for c in cat_cols]
	return df, cat_maps, cat_dims


def get_cont_mean_std(df: pd.DataFrame, cont_cols: List[str]) -> np.ndarray:
	stats = []
	for col in cont_cols:
		series = df[col].astype(float)
		mean = float(series.mean())
		std = float(series.std())
		if std < 1e-6:
			std = 1.0
		stats.append((mean, std))
	return np.array(stats, dtype=np.float32)


class MaskedTabularDataset(Dataset):
	def __init__(self, df: pd.DataFrame, cat_cols: List[str], cont_cols: List[str], mask_prob: float = 0.15):
		self.df = df.reset_index(drop=True)
		self.cat_cols = cat_cols
		self.cont_cols = cont_cols
		self.mask_prob = mask_prob
		self.mask_token = 0

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		x_categ = row[self.cat_cols].values.astype(int)
		x_cont = row[self.cont_cols].values.astype(float)
		mask = np.random.rand(len(x_categ)) < self.mask_prob

		# 保证每一行至少 mask 1 个字段，但不要全部 mask 掉。
		if len(mask) > 0 and not mask.any():
			mask[np.random.randint(len(mask))] = True
		if len(mask) > 1 and mask.all():
			mask[np.random.randint(len(mask))] = False

		x_categ_masked = x_categ.copy()
		x_categ_masked[mask] = self.mask_token

		return (
			torch.tensor(x_categ_masked, dtype=torch.long),
			torch.tensor(x_cont, dtype=torch.float32),
			torch.tensor(x_categ, dtype=torch.long),
			torch.tensor(mask, dtype=torch.bool),
		)


def train_epoch(model, loader, optimizer, criterion, cat_dims):
	model.train()
	total_loss = 0.0
	pbar = tqdm(loader, desc="Train", leave=False)

	for x_categ, x_cont, target, mask in pbar:
		x_categ = x_categ.to(DEVICE)
		x_cont = x_cont.to(DEVICE)
		target = target.to(DEVICE)
		mask = mask.to(DEVICE)

		logits = model(x_categ, x_cont)
		logits_split = torch.split(logits, cat_dims, dim=1)

		loss = 0.0
		n_masked = 0
		for logit_i, target_i, mask_i in zip(logits_split, target.T, mask.T):
			if mask_i.any():
				loss = loss + criterion(logit_i[mask_i], target_i[mask_i])
				n_masked += 1

		if n_masked > 0:
			loss = loss / n_masked
		else:
			loss = torch.tensor(0.0, device=DEVICE)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += float(loss.item())
		pbar.set_postfix({"loss": float(loss.item())})

	return total_loss / max(len(loader), 1)


def test_epoch(model, loader, criterion, cat_dims):
	model.eval()
	total_loss = 0.0

	with torch.no_grad():
		pbar = tqdm(loader, desc="Test", leave=False)
		for x_categ, x_cont, target, mask in pbar:
			x_categ = x_categ.to(DEVICE)
			x_cont = x_cont.to(DEVICE)
			target = target.to(DEVICE)
			mask = mask.to(DEVICE)

			logits = model(x_categ, x_cont)
			logits_split = torch.split(logits, cat_dims, dim=1)

			loss = 0.0
			n_masked = 0
			for logit_i, target_i, mask_i in zip(logits_split, target.T, mask.T):
				if mask_i.any():
					loss = loss + criterion(logit_i[mask_i], target_i[mask_i])
					n_masked += 1

			if n_masked > 0:
				loss = loss / n_masked
			else:
				loss = torch.tensor(0.0, device=DEVICE)

			total_loss += float(loss.item())
			pbar.set_postfix({"loss": float(loss.item())})

	return total_loss / max(len(loader), 1)


def main():
	print(f"从本地已下载数据读取训练集: {DATASET_ROOT}")
	print(f"DEVICE={DEVICE}")
	print(f"USE_PLAYER_IDENTITY={USE_PLAYER_IDENTITY}, USE_TEAM_IDENTITY={USE_TEAM_IDENTITY}")
	print(
		f"PRETRAIN_STRICT_PRIOR={PRETRAIN_STRICT_PRIOR}, "
		f"PRETRAIN_FULL_INFO={PRETRAIN_FULL_INFO}, CHECKPOINT={CHECKPOINT_PATH}"
	)
	raw_events = load_saved_events(
		save_root=DATASET_ROOT,
		limit_files=MAX_EVENT_FILES_FOR_TRAIN,
		max_rows=MAX_ROWS_FOR_TRAIN,
	)

	df, cat_cols, cont_cols = build_features(raw_events)
	df, cat_maps, cat_dims = encode_categories(df, cat_cols)

	cont_mean_std = get_cont_mean_std(df, cont_cols)

	dataset = MaskedTabularDataset(df, cat_cols, cont_cols, mask_prob=MASK_PROB)
	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

	model = TabTransformer(
		categories=tuple(cat_dims),
		num_continuous=len(cont_cols),
		dim=32,
		dim_out=sum(cat_dims),
		depth=6,
		heads=8,
		attn_dropout=0.1,
		ff_dropout=0.1,
		mlp_hidden_mults=(4, 2),
		mlp_act=nn.ReLU(),
		continuous_mean_std=torch.tensor(cont_mean_std, dtype=torch.float32),
	).to(DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	train_losses, test_losses = [], []
	for epoch in range(1, PRETRAIN_EPOCHS + 1):
		print(f"Epoch {epoch}:")
		train_loss = train_epoch(model, train_loader, optimizer, criterion, cat_dims)
		test_loss = test_epoch(model, test_loader, criterion, cat_dims)
		train_losses.append(train_loss)
		test_losses.append(test_loss)
		print(f"  Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"cat_cols": cat_cols,
			"cont_cols": cont_cols,
			"cat_maps": cat_maps,
			"cat_dims": cat_dims,
			"cont_mean_std": cont_mean_std,
			"use_player_identity": USE_PLAYER_IDENTITY,
			"use_team_identity": USE_TEAM_IDENTITY,
		},
		CHECKPOINT_PATH,
	)
	print(f"预训练权重已保存到: {CHECKPOINT_PATH}")

	plt.figure()
	plt.plot(train_losses, label="Train Loss")
	plt.plot(test_losses, label="Test Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.title("Row-wise Mask Reconstruction Pretraining")
	plt.savefig(LOSS_CURVE_OUTPUT)
	plt.close()
	print(f"Loss 曲线已保存到: {LOSS_CURVE_OUTPUT}")

	model.eval()
	with torch.no_grad():
		x_categ, x_cont, _, _ = dataset[0]
		x_categ = x_categ.unsqueeze(0).to(DEVICE)
		x_cont = x_cont.unsqueeze(0).to(DEVICE)
		embedding = model(x_categ, x_cont)
		print("Sample embedding shape:", embedding.shape)

	print(f"类别特征: {cat_cols}")
	print(f"连续特征: {cont_cols}")
	print(f"类别维度: {cat_dims}")
	print(f"类别映射字段数: {len(cat_maps)}")


if __name__ == "__main__":
	main()
