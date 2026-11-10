"""
eval_player_style.py

评估冻结预训练 TabEventEncoder 在球员风格表示任务上的能力。

两个评估任务：

1. **位置线性探针（Position Linear Probe）**
   每名球员 = 所有事件 embedding 的均值池化 → Logistic Regression 预测粗粒度位置
   粗粒度：GK / DEF / MID / ATT
   基线：直接用原始连续特征均值池化（无 encoder）

2. **同一球员跨赛场检索（Player Retrieval）**
   每名球员出场一次（一场比赛）= 一个 gallery 向量
   随机选若干出场为 query（另一场比赛同一球员）
   用余弦相似度检索，指标：Rank-1 Acc / MRR / Recall@5
   基线：同样用原始特征均值池化

重要：encoder 预训练时未使用 player/team identity，因此 embedding
代表的是动作风格，而非球员身份记忆。检索任务验证的是风格的跨场一致性。

用法：
  python eval_player_style.py
  python eval_player_style.py --max-matches 200   # 快速验证
  python eval_player_style.py --gpu
  python eval_player_style.py --min-appearances 5  # 只看出场>=5次的球员
"""

import argparse
import gc
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_xg_tabtransformer import (
    RAW_EVENT_COLUMNS,
    TabEventEncoder,
    build_event_features,
    set_seed,
    safe_text,
)
from train_vaep_baseline import (
    CACHE_DIR,
    SKIP_TYPES,
    TRAIN_ROOT,
    TEST_ROOT,
    list_event_files,
    safe_num,
)
from train_vaep_tabtransformer import load_pretrained_encoder

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEED = 42
PRETRAIN_CHECKPOINT = os.path.join("checkpoints", "tabtransformer_row_mask_pretrain.pt")
METRICS_OUTPUT = "player_style_eval_metrics.csv"

# StatsBomb position → 4-class label
POSITION_MAP = {
    # Goalkeeper
    "Goalkeeper": "GK",
    # Center backs
    "Center Back": "CB", "Right Center Back": "CB", "Left Center Back": "CB",
    # Wide defenders (full-backs + wing-backs)
    "Right Back": "WB", "Left Back": "WB", "Right Wing Back": "WB", "Left Wing Back": "WB",
    # Defensive / central midfield
    "Right Midfield": "DM", "Left Midfield": "DM", "Center Midfield": "DM",
    "Right Center Midfield": "DM", "Left Center Midfield": "DM",
    "Right Defensive Midfield": "DM", "Left Defensive Midfield": "DM",
    "Center Defensive Midfield": "DM",
    # Attacking midfield + wide forwards (wingers)
    "Right Attacking Midfield": "AM", "Left Attacking Midfield": "AM",
    "Center Attacking Midfield": "AM",
    "Right Wing": "AM", "Left Wing": "AM",
    # Central forwards
    "Right Center Forward": "FWD", "Left Center Forward": "FWD",
    "Center Forward": "FWD", "Secondary Striker": "FWD",
}
POSITION_CLASSES = ["GK", "CB", "WB", "DM", "AM", "FWD"]


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Player style evaluation via embedding probing")
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    p.add_argument("--max-matches", type=int, default=None,
                   help="Limit number of match files (for quick runs)")
    p.add_argument("--min-appearances", type=int, default=3,
                   help="Min match appearances per player for retrieval task")
    p.add_argument("--retrieval-queries-per-player", type=int, default=2,
                   help="How many query appearances per player in retrieval")
    p.add_argument("--embed-batch-size", type=int, default=2048)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--metrics-output", default=METRICS_OUTPUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction from one match file
# ---------------------------------------------------------------------------
def _get_position_label(raw_pos: str) -> Optional[str]:
    """Map raw StatsBomb position string to 4-class label."""
    return POSITION_MAP.get(raw_pos)


def extract_match_player_events(
    path: str,
    cat_cols: List[str],
    cont_cols: List[str],
) -> List[Dict]:
    """
    Read one parquet file, return list of records:
      {player, position_label, match_id, cat_feat, cont_feat}
    where cat_feat / cont_feat are raw string/float dicts (not yet encoded).
    Only includes rows where player AND position are known.
    """
    try:
        try:
            df = pd.read_parquet(path, columns=[c for c in RAW_EVENT_COLUMNS if c != "id"])
        except Exception:
            df = pd.read_parquet(path)
            keep = [c for c in RAW_EVENT_COLUMNS if c in df.columns]
            df = df[keep].copy()
    except Exception as e:
        print(f"  Skip {path}: {e}")
        return []

    if "player" not in df.columns or "position" not in df.columns:
        return []

    match_id = int(df["match_id"].iloc[0]) if "match_id" in df.columns else -1

    records = []
    for _, row in df.iterrows():
        etype = safe_text(row.get("type"))
        if etype in SKIP_TYPES:
            continue
        player = safe_text(row.get("player"))
        if player in ("__MISSING__", ""):
            continue
        raw_pos = safe_text(row.get("position"))
        pos_label = _get_position_label(raw_pos)
        if pos_label is None:
            continue

        team = safe_text(row.get("team"))
        possession_team = safe_text(row.get("possession_team"))
        seq_team = possession_team if possession_team != "__MISSING__" else team
        feat = build_event_features(row, cat_cols, cont_cols, sequence_team=seq_team)

        records.append({
            "player": player,
            "position_label": pos_label,
            "match_id": match_id,
            "feat": feat,
        })
    return records


# ---------------------------------------------------------------------------
# Build per-appearance (player, match) embedding and raw-feature profiles
# ---------------------------------------------------------------------------
def build_player_appearance_profiles(
    file_list: List[Tuple[str, str]],
    encoder: TabEventEncoder,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: Dict,
    device: torch.device,
    batch_size: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
        meta_df   : DataFrame with columns [player, position_label, match_id]
        emb_matrix: float32 [N_appearances, output_dim]  — encoder embeddings
        raw_matrix: float32 [N_appearances, len(cont_cols)] — raw feature mean
    One row = one (player, match) appearance = mean-pooled over all events.
    """
    # Accumulate per-appearance
    # key: (player, match_id)
    accum: Dict[Tuple[str, int], Dict] = defaultdict(
        lambda: {"pos_label": None, "cat_sum": None, "cont_sum": None, "n": 0,
                 "raw_sum": None}
    )

    encoder.eval()
    encoder = encoder.to(device)

    for path, src in tqdm(file_list, desc="Processing matches"):
        records = extract_match_player_events(path, cat_cols, cont_cols)
        if not records:
            continue

        n_total = len(records)

        # Build arrays for ALL events in this file at once → one encoder pass
        cat_arr  = np.zeros((n_total, len(cat_cols)),  dtype=np.int32)
        cont_arr = np.zeros((n_total, len(cont_cols)), dtype=np.float32)
        for i, r in enumerate(records):
            feat = r["feat"]
            for j, col in enumerate(cat_cols):
                cat_arr[i, j] = cat_maps[col].get(safe_text(feat.get(col)), 0)
            for j, col in enumerate(cont_cols):
                cont_arr[i, j] = safe_num(feat.get(col))

        # Single encoder pass for the whole file
        emb_chunks = []
        with torch.no_grad():
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                xc = torch.from_numpy(cat_arr[start:end]).to(device)
                xf = torch.from_numpy(cont_arr[start:end]).to(device)
                emb_chunks.append(encoder(xc, xf).cpu().float().numpy())
        emb_all = np.concatenate(emb_chunks, axis=0)  # [n_total, output_dim]

        # Group by (player, match_id) and mean-pool
        player_arr  = np.array([r["player"]         for r in records])
        matchid_arr = np.array([r["match_id"]        for r in records])
        poslabel_arr= np.array([r["position_label"]  for r in records])

        keys_here = set(zip(player_arr.tolist(), matchid_arr.tolist()))
        for (player, match_id) in keys_here:
            mask = (player_arr == player) & (matchid_arr == match_id)
            emb  = emb_all[mask].mean(axis=0)
            raw  = cont_arr[mask].mean(axis=0)
            pos  = poslabel_arr[mask][0]

            key = (player, match_id)
            acc = accum[key]
            if acc["cat_sum"] is None:
                acc["cat_sum"] = emb.copy()
                acc["raw_sum"] = raw.copy()
                acc["pos_label"] = pos
                acc["n"] = 1
            else:
                acc["cat_sum"] += emb
                acc["raw_sum"] += raw
                acc["n"] += 1

        del records, cat_arr, cont_arr, emb_all, emb_chunks
        gc.collect()

    # Assemble matrices
    keys = list(accum.keys())
    if not keys:
        return pd.DataFrame(), np.empty((0,)), np.empty((0,))

    # Infer dims
    sample = next(v for v in accum.values() if v["cat_sum"] is not None)
    emb_dim = len(sample["cat_sum"])
    raw_dim = len(sample["raw_sum"])

    emb_matrix = np.zeros((len(keys), emb_dim), dtype=np.float32)
    raw_matrix = np.zeros((len(keys), raw_dim), dtype=np.float32)
    meta_rows = []

    for i, (player, match_id) in enumerate(keys):
        acc = accum[(player, match_id)]
        emb_matrix[i] = acc["cat_sum"] / acc["n"]
        raw_matrix[i] = acc["raw_sum"] / acc["n"]
        meta_rows.append({
            "player": player,
            "match_id": match_id,
            "position_label": acc["pos_label"],
        })

    meta_df = pd.DataFrame(meta_rows)
    return meta_df, emb_matrix, raw_matrix


# ---------------------------------------------------------------------------
# Task 1: Position Linear Probe
# ---------------------------------------------------------------------------
def eval_position_probe(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    seed: int = SEED,
) -> List[Dict]:
    """
    Aggregate per player (mean over all appearances), then run Logistic Regression.
    Train on players from first half of match_ids, test on second half.
    """
    print("\n=== Task 1: Position Linear Probe ===")

    # Aggregate per player
    players = meta_df["player"].unique()
    player_embs = []
    player_raws = []
    player_positions = []

    for pl in players:
        mask = meta_df["player"] == pl
        pos_labels = meta_df.loc[mask, "position_label"].dropna().unique()
        if len(pos_labels) == 0:
            continue
        # Majority vote for position
        from collections import Counter
        pos = Counter(meta_df.loc[mask, "position_label"].dropna()).most_common(1)[0][0]
        player_embs.append(emb_matrix[mask].mean(axis=0))
        player_raws.append(raw_matrix[mask].mean(axis=0))
        player_positions.append(pos)

    X_emb = np.array(player_embs, dtype=np.float32)
    X_raw = np.array(player_raws, dtype=np.float32)
    y = np.array(player_positions)

    if len(X_emb) < 20:
        print("  Not enough players for probe, skipping.")
        return []

    # Shuffle and split 70/30
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_emb))
    split = int(len(idx) * 0.7)
    tr, te = idx[:split], idx[split:]

    results = []
    for name, X in [("encoder_embedding", X_emb), ("raw_cont_features", X_raw)]:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0,
                                 class_weight="balanced", solver="lbfgs",
                                 multi_class="multinomial")
        clf.fit(Xtr, y[tr])
        preds = clf.predict(Xte)
        acc = float(accuracy_score(y[te], preds))
        print(f"\n  [{name}]  Accuracy = {acc:.3f}  (n_train={len(tr)}, n_test={len(te)})")
        print(classification_report(y[te], preds, labels=POSITION_CLASSES, zero_division=0))
        results.append({
            "task": "position_probe",
            "features": name,
            "accuracy": acc,
            "n_players": len(X_emb),
            "n_train": len(tr),
            "n_test": len(te),
        })
    return results


# ---------------------------------------------------------------------------
# Task 2: Player Retrieval
# ---------------------------------------------------------------------------
def _cosine_sim_matrix(Q: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix [n_queries, n_gallery]."""
    Q_n = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    G_n = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return Q_n @ G_n.T


def eval_player_retrieval(
    meta_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    min_appearances: int = 3,
    queries_per_player: int = 2,
    seed: int = SEED,
    within_position: bool = False,
) -> List[Dict]:
    """
    For each player with >= min_appearances:
      - Pick `queries_per_player` appearances as query
      - Remaining appearances form the gallery (one per player, mean-pooled)
      - Retrieve correct player by cosine similarity
    when within_position=True, gallery is restricted to players of the same position.
    Metrics: Rank-1, MRR, Recall@5/10/20
    """
    mode_label = "within-position" if within_position else "global"
    print(f"\n=== Task 2: Player Retrieval ({mode_label}) ===")

    rng = np.random.default_rng(seed)

    # Count appearances per player
    counts = meta_df["player"].value_counts()
    eligible = counts[counts >= min_appearances].index.tolist()
    print(f"  Eligible players (>= {min_appearances} appearances): {len(eligible)}")

    if len(eligible) < 10:
        print("  Not enough eligible players, skipping.")
        return []

    # Majority-vote position per player (for within-position mode)
    player_position: Dict[str, str] = {}
    for pl in eligible:
        pl_mask = meta_df["player"].values == pl
        player_position[pl] = meta_df.loc[pl_mask, "position_label"].mode()[0]

    results = []
    for feat_name, matrix in [("encoder_embedding", emb_matrix),
                               ("raw_cont_features", raw_matrix)]:
        queries_feat = []
        queries_player_id: List[str] = []
        queries_position: List[str] = []
        gallery_feat = []
        gallery_player_id: List[str] = []
        gallery_position_arr: List[str] = []

        for pl in eligible:
            pl_idx = np.where(meta_df["player"].values == pl)[0]
            perm = rng.permutation(len(pl_idx))
            q_count = min(queries_per_player, len(pl_idx) - 1)
            q_idx = pl_idx[perm[:q_count]]
            g_idx = pl_idx[perm[q_count:]]

            for qi in q_idx:
                queries_feat.append(matrix[qi])
                queries_player_id.append(pl)
                queries_position.append(player_position[pl])

            gallery_feat.append(matrix[g_idx].mean(axis=0))
            gallery_player_id.append(pl)
            gallery_position_arr.append(player_position[pl])

        Q = np.array(queries_feat, dtype=np.float32)        # [n_q, D]
        G = np.array(gallery_feat, dtype=np.float32)        # [n_g, D]
        gal_ids  = np.array(gallery_player_id)
        gal_pos  = np.array(gallery_position_arr)

        sim = _cosine_sim_matrix(Q, G)   # [n_q, n_g]

        rank1_hits = mrr_sum = r5_hits = r10_hits = r20_hits = 0
        n_q_counted = 0

        for qi, (true_pl, true_pos) in enumerate(zip(queries_player_id, queries_position)):
            if within_position:
                candidates = np.where(gal_pos == true_pos)[0]
                if len(candidates) < 2:   # need at least one other player
                    continue
                ranked = candidates[np.argsort(-sim[qi][candidates])]
            else:
                ranked = np.argsort(-sim[qi])

            n_q_counted += 1
            for rank, gi in enumerate(ranked):
                if gal_ids[gi] == true_pl:
                    if rank == 0:  rank1_hits += 1
                    mrr_sum += 1.0 / (rank + 1)
                    if rank <  5:  r5_hits  += 1
                    if rank < 10:  r10_hits += 1
                    if rank < 20:  r20_hits += 1
                    break

        if n_q_counted == 0:
            continue

        rank1 = rank1_hits / n_q_counted
        mrr   = mrr_sum    / n_q_counted
        r5    = r5_hits    / n_q_counted
        r10   = r10_hits   / n_q_counted
        r20   = r20_hits   / n_q_counted

        if within_position:
            from collections import Counter
            pos_sizes = Counter(gallery_position_arr)
            avg_gal = np.mean(list(pos_sizes.values()))
            gal_info = f"avg {avg_gal:.0f} per position ({len(G)} total eligible)"
        else:
            gal_info = str(len(G))

        print(f"\n  [{feat_name}]")
        print(f"    n_queries={n_q_counted}, n_gallery={gal_info}")
        print(f"    Rank-1 = {rank1:.3f}  |  MRR = {mrr:.3f}  "
              f"|  R@5 = {r5:.3f}  |  R@10 = {r10:.3f}  |  R@20 = {r20:.3f}")
        results.append({
            "task": f"player_retrieval_{mode_label}",
            "features": feat_name,
            "rank1_acc": rank1,
            "mrr": mrr,
            "recall_at_5":  r5,
            "recall_at_10": r10,
            "recall_at_20": r20,
            "n_queries":  n_q_counted,
            "n_gallery":  len(G),
        })

    return results


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"DEVICE = {device}")

    # Load encoder
    print(f"Loading encoder from {args.pretrain_checkpoint} ...")
    encoder, cat_cols, cont_cols, cat_maps, _, _ = \
        load_pretrained_encoder(args.pretrain_checkpoint)
    print(f"Encoder output_dim = {encoder.output_dim}")
    print(f"Note: encoder was pretrained WITHOUT player/team identity — "
          f"embeddings represent action style, not player ID.")

    # Collect all event files (train + test combined for more players)
    train_files = list_event_files(TRAIN_ROOT, max_match_files=args.max_matches)
    test_files  = list_event_files(TEST_ROOT,  max_match_files=args.max_matches)
    all_files   = train_files + test_files
    print(f"Total match files: {len(all_files)}")

    # Build player appearance profiles
    print("\nBuilding player appearance profiles (mean-pooled embeddings per match) ...")
    meta_df, emb_matrix, raw_matrix = build_player_appearance_profiles(
        all_files, encoder, cat_cols, cont_cols, cat_maps,
        device, args.embed_batch_size,
    )

    print(f"\nTotal appearances (player × match): {len(meta_df)}")
    print(f"Unique players: {meta_df['player'].nunique()}")
    print(f"Position distribution:\n{meta_df['position_label'].value_counts().to_string()}")

    # Run tasks
    all_results = []

    probe_results = eval_position_probe(meta_df, emb_matrix, raw_matrix)
    all_results.extend(probe_results)

    retrieval_results = eval_player_retrieval(
        meta_df, emb_matrix, raw_matrix,
        min_appearances=args.min_appearances,
        queries_per_player=args.retrieval_queries_per_player,
        within_position=False,
    )
    all_results.extend(retrieval_results)

    retrieval_within_results = eval_player_retrieval(
        meta_df, emb_matrix, raw_matrix,
        min_appearances=args.min_appearances,
        queries_per_player=args.retrieval_queries_per_player,
        within_position=True,
    )
    all_results.extend(retrieval_within_results)

    # Save metrics
    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(args.metrics_output, index=False, encoding="utf-8-sig")
        print(f"\nSaved metrics → {args.metrics_output}")
        print(df_out.to_string(index=False))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
