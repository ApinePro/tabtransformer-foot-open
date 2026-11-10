import os
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from statsbombpy import sb
except ImportError as exc:
    raise ImportError(
        "statsbombpy 未安装，请先执行: pip install statsbombpy pyarrow"
    ) from exc


TRAIN_SAVE_ROOT = "dataset"
TEST_SAVE_ROOT = "dataset-test"
MAX_MATCHES_PER_SEASON: Optional[int] = None  # None = 下载该赛季全部比赛
SAVE_CSV = False  # True 会额外保存 csv，但会占更多空间
REFRESH = False
DOWNLOAD_EVENTS = True
DOWNLOAD_LINEUPS = True
DOWNLOAD_360 = True

# 训练集：2015/2016 赛季五大联赛
TRAIN_TARGET_SEASON_NAME: Optional[str] = "2015/2016"
TRAIN_TARGET_COMPETITIONS = [
    "Premier League",
    "La Liga",
    "Serie A",
    "1. Bundesliga",
    "Ligue 1",
]

# 论文/评估常用的独立测试集，单独下载到 dataset-test/，避免泄漏进训练集。
TEST_TARGETS = [
    {"competition_name": "UEFA Euro", "season_name": "2020", "country_name": "Europe"},
    {"competition_name": "FIFA World Cup", "season_name": "2022", "country_name": "International"},
    {"competition_name": "Copa America", "season_name": "2024", "country_name": "South America"},
]


def sanitize_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "unknown"


def save_dataframe(df: pd.DataFrame, path_without_ext: str, save_csv: bool = SAVE_CSV) -> None:
    directory = os.path.dirname(path_without_ext)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(f"{path_without_ext}.parquet", index=False)
    if save_csv:
        df.to_csv(f"{path_without_ext}.csv", index=False)


def normalize_lineups(lineups, match_id: int) -> pd.DataFrame:
    if not isinstance(lineups, dict):
        return pd.DataFrame()

    frames = []
    for team_name, team_df in lineups.items():
        if isinstance(team_df, pd.DataFrame) and len(team_df) > 0:
            tmp = team_df.copy()
            tmp["team_name"] = team_name
            tmp["match_id"] = match_id
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_competitions(
    competitions: pd.DataFrame,
    target_season_name: Optional[str] = TRAIN_TARGET_SEASON_NAME,
    target_competitions=None,
    target_pairs=None,
) -> pd.DataFrame:
    comp_view = competitions[[
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
    ]].drop_duplicates()

    if target_pairs:
        mask = pd.Series(False, index=comp_view.index, dtype=bool)
        for pair in target_pairs:
            pair_mask = comp_view["competition_name"].eq(pair["competition_name"])
            if pair.get("season_name") is not None:
                pair_mask &= comp_view["season_name"].eq(pair["season_name"])
            if pair.get("country_name") is not None:
                pair_mask &= comp_view["country_name"].eq(pair["country_name"])
            mask |= pair_mask
        comp_view = comp_view[mask]
    else:
        if target_season_name is not None:
            comp_view = comp_view[comp_view["season_name"] == target_season_name]

        if target_competitions:
            comp_view = comp_view[comp_view["competition_name"].isin(target_competitions)]

    return comp_view.reset_index(drop=True)


def download_all_open_data(
    save_root: str = TRAIN_SAVE_ROOT,
    max_matches_per_season: Optional[int] = MAX_MATCHES_PER_SEASON,
    refresh: bool = REFRESH,
    save_csv: bool = SAVE_CSV,
    target_season_name: Optional[str] = TRAIN_TARGET_SEASON_NAME,
    target_competitions=TRAIN_TARGET_COMPETITIONS,
    target_pairs=None,
) -> pd.DataFrame:
    os.makedirs(save_root, exist_ok=True)

    print("拉取 competitions...")
    competitions = sb.competitions()
    save_dataframe(competitions, os.path.join(save_root, "competitions"), save_csv=save_csv)

    manifest_rows = []
    total_event_rows = 0
    total_saved_matches = 0

    comp_view = filter_competitions(
        competitions,
        target_season_name=target_season_name,
        target_competitions=target_competitions,
        target_pairs=target_pairs,
    )

    if len(comp_view) == 0:
        raise ValueError(
            f"没有找到符合条件的赛事: season={target_season_name}, competitions={target_competitions}, target_pairs={target_pairs}"
        )

    save_dataframe(comp_view, os.path.join(save_root, "selected_competitions"), save_csv=True)
    print("本次将下载以下赛事/赛季:")
    print(comp_view.to_string(index=False))

    for _, row in tqdm(comp_view.iterrows(), total=len(comp_view), desc="Competitions/Seasons"):
        competition_id = int(row["competition_id"])
        season_id = int(row["season_id"])
        season_dir = os.path.join(
            save_root,
            f"comp_{competition_id}_season_{season_id}_{sanitize_name(row['country_name'])}_{sanitize_name(row['competition_name'])}_{sanitize_name(row['season_name'])}",
        )

        matches_path = os.path.join(season_dir, "matches")
        if os.path.exists(f"{matches_path}.parquet") and not refresh:
            matches = pd.read_parquet(f"{matches_path}.parquet")
        else:
            matches = sb.matches(competition_id=competition_id, season_id=season_id)
            save_dataframe(matches, matches_path, save_csv=save_csv)

        match_ids = matches["match_id"].tolist()
        if max_matches_per_season is not None:
            match_ids = match_ids[:max_matches_per_season]

        for match_id in tqdm(match_ids, desc=f"matches {competition_id}-{season_id}", leave=False):
            item = {
                "competition_id": competition_id,
                "season_id": season_id,
                "match_id": int(match_id),
                "events_saved": False,
                "lineups_saved": False,
                "frames_saved": False,
                "event_rows": 0,
            }

            if DOWNLOAD_EVENTS:
                try:
                    events_path = os.path.join(season_dir, "events", str(match_id))
                    if os.path.exists(f"{events_path}.parquet") and not refresh:
                        events = pd.read_parquet(f"{events_path}.parquet")
                    else:
                        events = sb.events(match_id=match_id, split=False, flatten_attrs=True)
                        if isinstance(events, pd.DataFrame) and len(events) > 0:
                            save_dataframe(events, events_path, save_csv=save_csv)

                    if isinstance(events, pd.DataFrame) and len(events) > 0:
                        item["events_saved"] = True
                        item["event_rows"] = int(len(events))
                        total_event_rows += int(len(events))
                except Exception as exc:
                    item["events_error"] = str(exc)

            if DOWNLOAD_LINEUPS:
                try:
                    lineups = sb.lineups(match_id=match_id)
                    lineups_df = normalize_lineups(lineups, match_id)
                    if len(lineups_df) > 0:
                        save_dataframe(
                            lineups_df,
                            os.path.join(season_dir, "lineups", str(match_id)),
                            save_csv=save_csv,
                        )
                        item["lineups_saved"] = True
                except Exception as exc:
                    item["lineups_error"] = str(exc)

            if DOWNLOAD_360:
                try:
                    frames = sb.frames(match_id=match_id)
                    if isinstance(frames, pd.DataFrame) and len(frames) > 0:
                        save_dataframe(
                            frames,
                            os.path.join(season_dir, "frames", str(match_id)),
                            save_csv=save_csv,
                        )
                        item["frames_saved"] = True
                except Exception as exc:
                    item["frames_error"] = str(exc)

            total_saved_matches += 1
            manifest_rows.append(item)

    manifest = pd.DataFrame(manifest_rows)
    save_dataframe(manifest, os.path.join(save_root, "download_manifest"), save_csv=True)

    print("下载完成")
    print(f"保存目录: {save_root}")
    print(f"处理比赛数: {total_saved_matches}")
    print(f"累计事件行数: {total_event_rows}")
    if save_root == TRAIN_SAVE_ROOT:
        print("你可以在 main.py 中直接从 dataset/ 读取训练。")
    elif save_root == TEST_SAVE_ROOT:
        print("dataset-test/ 是独立测试集，不建议混入 main.py 的训练数据。")
    return manifest


if __name__ == "__main__":
    print("=== 下载训练集：2015/2016 五大联赛 ===")
    download_all_open_data(
        save_root=TRAIN_SAVE_ROOT,
        target_season_name=TRAIN_TARGET_SEASON_NAME,
        target_competitions=TRAIN_TARGET_COMPETITIONS,
    )

    print("\n=== 下载独立测试集：Euro 2020 / World Cup 2022 / Copa America 2024 ===")
    download_all_open_data(
        save_root=TEST_SAVE_ROOT,
        target_season_name=None,
        target_competitions=None,
        target_pairs=TEST_TARGETS,
    )
