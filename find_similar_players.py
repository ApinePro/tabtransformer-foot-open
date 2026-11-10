"""
find_similar_players.py
=======================
Find the most action-style-similar players using the frozen TabEventEncoder.

Two built-in demo tasks
  --demo elite        iconic players in 2015/16 — how do they cluster?
  --demo young-stars  rising talents in 2015/16 — who do they play like?

Player gallery is cached to .vaep_cache/ after the first run (~35 min on GPU).
Subsequent runs load in seconds.

Examples:
  python find_similar_players.py --gpu
  python find_similar_players.py --gpu --demo young-stars --top-k 10
  python find_similar_players.py --gpu --queries "Lionel Messi" "Kevin De Bruyne"
  python find_similar_players.py --gpu --max-matches 200   # quick test
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from train_vaep_baseline import list_event_files, TRAIN_ROOT, TEST_ROOT
from train_vaep_tabtransformer import load_pretrained_encoder
from eval_player_style import (
    build_player_appearance_profiles,
    _cosine_sim_matrix,
    PRETRAIN_CHECKPOINT,
)

# ── defaults ──────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(".vaep_cache")
OUTPUT_CSV  = "similar_players_results.csv"
TOP_K       = 10
MIN_GALLERY_APPEARANCES = 5   # players with fewer appearances are excluded from gallery
EMBED_BATCH_SIZE = 2048
BIRTHDATE_TABLE = "train_players_birthdate_table.csv"
AGE_REF_DATE    = "2016-06-01"   # season midpoint for the 2015/16 training data

# ── built-in demo queries ─────────────────────────────────────────────────────
DEMO_QUERIES: Dict[str, List[str]] = {
    "elite": [
        # Forwards
        "Messi",
        "Cristiano Ronaldo",
        "Lewandowski",
        "Suárez",
        "Higuaín",
        "Cavani",
        # Creative midfielders
        "De Bruyne",
        "Iniesta",
        "Modrić",
        "Kroos",
        # Defenders
        "Sergio Ramos",
        "Thiago Silva",
    ],
    "young-stars": [
        # Premier League
        "Dele Alli",          # Tottenham · PL · age 19
        "Marcus Rashford",    # Man United · PL · age 18 · debuted Feb 2016
        "Anthony Martial",    # Man United · PL · age 19
        "Raheem Sterling",    # Man City   · PL · age 21
        # La Liga
        "Paulo Dybala",       # Juventus   · SA · age 22 · breakout season (via Italy below)
        "Marco Asensio",      # Espanyol   · La Liga · age 20 (loan from Real Madrid)
        # Serie A
        "Gianluigi Donnarumma",  # AC Milan · SA · age 16-17 · debuted Oct 2015
        # Bundesliga
        "Leroy Sané",         # Schalke    · Bundesliga · age 19
        "Timo Werner",        # Stuttgart  · Bundesliga · age 19
        "Kingsley Coman",     # Bayern München · Bundesliga · age 19 (loan from Juventus)
        "Julian Weigl",       # Dortmund   · Bundesliga · age 20
        # Ligue 1
        "Kylian Mbappé",      # Monaco     · Ligue 1 · age 17 · debuted Dec 2015
        "Ousmane Dembélé",    # Rennes     · Ligue 1 · age 18
        "Presnel Kimpembe",   # PSG        · Ligue 1 · age 20
    ],
    "next-gen": [
        # Defenders who became world-class
        "Raphaël Varane",     # Real Madrid · La Liga · age 22
        "Samuel Umtiti",      # Lyon        · Ligue 1 · age 21
        "Aymeric Laporte",    # Athletic Club · La Liga · age 21
        "Lucas Digne",        # PSG         · Ligue 1 · age 22
        "Benjamin Mendy",     # Monaco      · Ligue 1 · age 21
        "João Cancelo",       # Valencia    · La Liga · age 21
        "Benjamin Pavard",    # Stuttgart   · Bundesliga · age 19
        # Midfield playmakers
        "Marco Verratti",     # PSG         · Ligue 1 · age 22
        "Adrien Rabiot",      # PSG         · Ligue 1 · age 20
        "Geoffrey Kondogbia", # Inter       · Serie A · age 22
        "Christian Pulisic",  # Dortmund    · Bundesliga · age 17 · debuted Jan 2016
        # Attackers
        "Nabil Fekir",        # Lyon        · Ligue 1 · age 22
        "Wissam Ben Yedder",  # Sevilla     · La Liga · age 25 · breakout
        "Maxwell Cornet",     # Lyon        · Ligue 1 · age 19
        "Jean-Kévin Augustin",# PSG         · Ligue 1 · age 18
        "Christopher Nkunku", # PSG         · Ligue 1 · age 18 · early appearances
        "Jack Grealish",      # Aston Villa · PL · age 20
        "Malcom",             # Bordeaux    · Ligue 1 · age 18
    ],
    "u20": [
        # La Liga
        "Mikel Oyarzabal",    # Real Sociedad · La Liga · age 18
        "Daniel Ceballos",    # Real Betis    · La Liga · age 19
        "Théo Bongonda",      # Celta Vigo    · La Liga · age 19 (loan)
        # Bundesliga
        "Julian Brandt",      # Leverkusen    · Bundesliga · age 19
        "Jonathan Tah",       # Leverkusen    · Bundesliga · age 19 (CB)
        # Serie A
        "Lorenzo Pellegrini", # Sassuolo      · Serie A · age 18 (loan from Roma)
        # Premier League
        "Ruben Loftus-Cheek", # Chelsea       · PL · age 19
    ],
    "chelsea": [
        # First team youngsters
        "Kurt Zouma",         # Chelsea · PL · age 20 · CB
        "Kenedy",             # Chelsea · PL · age 19 · Brazilian winger
        "Charly Musonda",     # Real Betis loan · La Liga · age 18
        # External loanees (data from their loan clubs)
        "Andreas Christensen",# Gladbach · Bundesliga · age 19 · CB
        "Patrick Bamford",    # Crystal Palace / MU loan · PL · age 21
        # First team veterans for comparison
        "Eden Hazard",        # Chelsea · PL · age 24
        "Francesc Fàbregas",  # Chelsea · PL · age 28
        "Oscar dos Santos",   # Chelsea · PL · age 24
    ],
}

# Context notes shown alongside young-star query headers
PLAYER_CONTEXT: Dict[str, str] = {
    # Premier League
    "dele alli":             "Tottenham · PL 2015/16 · age 19 · breakthrough season",
    "marcus rashford":       "Man United · PL 2015/16 · age 18 · debuted Feb 2016",
    "anthony martial":       "Man United · PL 2015/16 · age 19 · signed from Monaco Jul 2015",
    "raheem sterling":       "Man City · PL 2015/16 · age 21 · signed from Liverpool Jul 2015",
    # La Liga / Serie A
    "paulo dybala":          "Juventus · Serie A 2015/16 · age 22 · breakout season",
    "marco asensio":         "Espanyol (loan) · La Liga 2015/16 · age 20",
    # Serie A
    "gianluigi donnarumma": "AC Milan · Serie A 2015/16 · age 16-17 · debuted Oct 2015",
    # Bundesliga
    "leroy sané":            "Schalke · Bundesliga 2015/16 · age 19",
    "timo werner":           "VfB Stuttgart · Bundesliga 2015/16 · age 19",
    "kingsley coman":        "Bayern München · Bundesliga 2015/16 · age 19 (loan from Juventus)",
    "julian weigl":          "Dortmund · Bundesliga 2015/16 · age 20",
    # Ligue 1
    "kylian mbappé":         "Monaco · Ligue 1 2015/16 · age 17 · debuted Dec 2015",
    "ousmane dembélé":       "Rennes · Ligue 1 2015/16 · age 18",
    "presnel kimpembe":      "PSG · Ligue 1 2015/16 · age 20",
    # next-gen
    "raphaël varane":        "Real Madrid · La Liga 2015/16 · age 22",
    "samuel yves umtiti":    "Lyon · Ligue 1 2015/16 · age 21",
    "aymeric laporte":       "Athletic Club · La Liga 2015/16 · age 21",
    "lucas digne":           "PSG · Ligue 1 2015/16 · age 22",
    "benjamin mendy":        "Monaco · Ligue 1 2015/16 · age 21",
    "joão pedro cavaco cancelo": "Valencia · La Liga 2015/16 · age 21",
    "benjamin pavard":       "VfB Stuttgart · Bundesliga 2015/16 · age 19",
    "marco verratti":        "PSG · Ligue 1 2015/16 · age 22",
    "adrien rabiot":         "PSG · Ligue 1 2015/16 · age 20",
    "geoffrey kondogbia":    "Inter · Serie A 2015/16 · age 22",
    "christian pulisic":     "Dortmund · Bundesliga 2015/16 · age 17 · debuted Jan 2016",
    "nabil fekir":           "Lyon · Ligue 1 2015/16 · age 22",
    "wissam ben yedder":     "Sevilla · La Liga 2015/16 · age 25 · breakout season",
    "gnaly maxwell cornet":  "Lyon · Ligue 1 2015/16 · age 19",
    "jean-kévin augustin":   "PSG · Ligue 1 2015/16 · age 18",
    "christopher nkunku":    "PSG · Ligue 1 2015/16 · age 18 · early appearances",
    "jack grealish":         "Aston Villa · PL 2015/16 · age 20",
    "malcom filipe silva de oliveira": "Bordeaux · Ligue 1 2015/16 · age 18",
    # u20
    "mikel oyarzabal ugarte":     "Real Sociedad · La Liga 2015/16 · age 18",
    "daniel ceballos fernández":  "Real Betis · La Liga 2015/16 · age 19 · iniesta-style DM",
    "théo bongonda mbul'ofeko batombo": "Celta Vigo · La Liga 2015/16 · age 19 (loan)",
    "julian brandt":              "Leverkusen · Bundesliga 2015/16 · age 19",
    "jonathan tah":               "Leverkusen · Bundesliga 2015/16 · age 19 · CB",
    "lorenzo pellegrini":         "Sassuolo · Serie A 2015/16 · age 18 (loan from Roma)",
    "ruben loftus-cheek":         "Chelsea · PL 2015/16 · age 19",
    # chelsea
    "kurt happy zouma":           "Chelsea · PL 2015/16 · age 20 · CB (joined from St-Étienne 2014)",
    "robert kenedy nunes do nascimento": "Chelsea · PL 2015/16 · age 19 · Brazilian winger",
    "charly musonda junior":       "Real Betis (loan) · La Liga 2015/16 · age 18 · dribbler",
    "andreas christensen":        "Gladbach (loan) · Bundesliga 2015/16 · age 19 · CB",
    "patrick bamford":            "Crystal Palace (loan) · PL 2015/16 · age 21",
    "eden hazard":                "Chelsea · PL 2015/16 · age 24 · PFA Player of the Year 2015",
    "francesc fàbregas i soler":   "Chelsea · PL 2015/16 · age 28",
    "oscar dos santos emboaba júnior": "Chelsea · PL 2015/16 · age 24",
}


# ── gallery building + caching ────────────────────────────────────────────────
def _gallery_cache_key(ckpt_path: str, files: List[Tuple[str, str]]) -> str:
    content = ckpt_path + "|" + "|".join(sorted(p for p, _ in files))
    return hashlib.md5(content.encode()).hexdigest()[:12]


def build_player_gallery(
    encoder,
    cat_cols: List[str],
    cont_cols: List[str],
    cat_maps: Dict,
    files: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int,
    ckpt_path: str,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (or load from cache) per-player aggregated embeddings.

    Returns
    -------
    names      : str [N]        player names
    positions  : str [N]        majority-vote position label
    n_apps     : int [N]        number of appearances (match files)
    emb_matrix : float32 [N,D]  mean-pooled encoder embedding
    raw_matrix : float32 [N,C]  mean-pooled raw continuous features
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"player_gallery_{_gallery_cache_key(ckpt_path, files)}.npz"

    if use_cache and cache_file.exists():
        print(f"Loading player gallery from cache: {cache_file}")
        d = np.load(cache_file, allow_pickle=True)
        return d["names"], d["positions"], d["n_apps"], d["emb_matrix"], d["raw_matrix"]

    print("Building player gallery — this takes ~35 min on GPU and will be cached ...")
    meta_df, emb_per_app, raw_per_app = build_player_appearance_profiles(
        files, encoder, cat_cols, cont_cols, cat_maps, device, batch_size,
    )

    # Aggregate per player (mean over all appearances)
    names_list:    List[str]       = []
    positions_list: List[str]      = []
    n_apps_list:   List[int]       = []
    emb_list:      List[np.ndarray] = []
    raw_list:      List[np.ndarray] = []

    for player in meta_df["player"].unique():
        mask = (meta_df["player"] == player).values
        pos_counts = Counter(
            lbl for lbl in meta_df.loc[mask, "position_label"] if pd.notna(lbl)
        )
        if not pos_counts:
            continue
        names_list.append(player)
        positions_list.append(pos_counts.most_common(1)[0][0])
        n_apps_list.append(int(mask.sum()))
        emb_list.append(emb_per_app[mask].mean(axis=0))
        raw_list.append(raw_per_app[mask].mean(axis=0))

    names      = np.array(names_list)
    positions  = np.array(positions_list)
    n_apps     = np.array(n_apps_list, dtype=np.int32)
    emb_matrix = np.array(emb_list, dtype=np.float32)
    raw_matrix = np.array(raw_list, dtype=np.float32)

    np.savez_compressed(
        cache_file,
        names=names, positions=positions, n_apps=n_apps,
        emb_matrix=emb_matrix, raw_matrix=raw_matrix,
    )
    print(f"Gallery cached → {cache_file}  ({len(names)} players)")
    return names, positions, n_apps, emb_matrix, raw_matrix


# ── query resolution ──────────────────────────────────────────────────────────
def resolve_query(query: str, names: np.ndarray) -> Optional[int]:
    """
    Case-insensitive search.  Priority order:
    1. Exact match.
    2. Full query is a substring of the gallery name.
    3. All words in the query appear (anywhere) in the gallery name
       — handles "Lionel Messi" → "Lionel Andrés Messi Cuccittini",
         "Luis Suárez" → "Luis Alberto Suárez Díaz", etc.
    4. Ambiguous → pick the gallery name with highest word-overlap ratio.
    """
    q = query.lower().strip()
    names_lower = [n.lower() for n in names]

    # 1. exact
    for i, nl in enumerate(names_lower):
        if nl == q:
            return int(i)

    # 2. full substring
    hits = [i for i, nl in enumerate(names_lower) if q in nl]
    if len(hits) == 1:
        return int(hits[0])
    if len(hits) > 1:
        return int(max(hits, key=lambda i: len(q) / max(len(names_lower[i]), 1)))

    # 3. all-words match
    words = q.split()
    hits = [i for i, nl in enumerate(names_lower) if all(w in nl for w in words)]
    if len(hits) == 0:
        return None
    if len(hits) == 1:
        return int(hits[0])
    # ambiguous → most words covered relative to name length
    return int(max(hits, key=lambda i: sum(w in names_lower[i] for w in words) / max(len(names_lower[i]), 1)))


# ── age helper ────────────────────────────────────────────────────────────────
def _build_age_array(
    names: np.ndarray,
    birthdate_csv: str,
    ref_date: str,
) -> np.ndarray:
    """Return float age (years) for each gallery name at ref_date; NaN if unknown."""
    from datetime import datetime

    try:
        bd = pd.read_csv(birthdate_csv, dtype=str).fillna("")
    except FileNotFoundError:
        print(f"[warn] birthdate table not found: {birthdate_csv} — age filter disabled")
        return np.full(len(names), np.nan, dtype=np.float32)

    name_to_dob: Dict[str, str] = {}
    for _, r in bd.iterrows():
        nm = str(r.get("player_name", "")).strip().lower()
        dob = str(r.get("birth_date", "")).strip()
        if nm and dob and dob not in name_to_dob.get(nm, ""):
            name_to_dob[nm] = dob

    ref = datetime.strptime(ref_date, "%Y-%m-%d")
    ages = np.full(len(names), np.nan, dtype=np.float32)
    n_match = 0
    for i, name in enumerate(names):
        dob = name_to_dob.get(str(name).strip().lower(), "")
        if not dob:
            continue
        try:
            d = datetime.strptime(dob, "%Y-%m-%d")
        except Exception:
            continue
        ages[i] = (ref - d).days / 365.25
        n_match += 1
    print(f"Birthdate match: {n_match}/{len(names)} gallery players resolved (ref={ref_date})")
    return ages


# ── retrieval ─────────────────────────────────────────────────────────────────
def find_similar(
    query_idx: int,
    names: np.ndarray,
    positions: np.ndarray,
    n_apps: np.ndarray,
    emb_matrix: np.ndarray,
    top_k: int,
    min_gallery_appearances: int,
    extra_eligible: Optional[np.ndarray] = None,
    ages: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Cosine similarity retrieval; query player excluded from gallery.

    Optional `extra_eligible` boolean mask further restricts the gallery
    (e.g. U21 filter).  The query player itself is always excluded but is
    *not* required to satisfy `extra_eligible`.
    """
    eligible = (n_apps >= min_gallery_appearances) & (np.arange(len(names)) != query_idx)
    if extra_eligible is not None:
        eligible = eligible & extra_eligible
    g_idx = np.where(eligible)[0]

    if g_idx.size == 0:
        return []

    sims = _cosine_sim_matrix(emb_matrix[[query_idx]], emb_matrix[g_idx])[0]
    ranked = np.argsort(-sims)[:top_k]

    out = []
    for r in ranked:
        gi = int(g_idx[r])
        row = {
            "rank":        int(r + 1),
            "player":      str(names[gi]),
            "position":    str(positions[gi]),
            "similarity":  float(sims[r]),
            "appearances": int(n_apps[gi]),
        }
        if ages is not None and not np.isnan(ages[gi]):
            row["age"] = round(float(ages[gi]), 1)
        out.append(row)
    return out


# ── formatted output ──────────────────────────────────────────────────────────
_BAR = "─" * 72

def print_results(
    query_name: str,
    query_pos: str,
    query_apps: int,
    similar: List[Dict],
    context_note: str = "",
) -> None:
    print(f"\n{_BAR}")
    ctx = f"  ▸ {context_note}" if context_note else ""
    print(f"  {query_name}  [{query_pos} | {query_apps} appearances]{ctx}")
    print(_BAR)
    print(f"  {'#':>3}  {'Player':<34}  {'Pos':>4}  {'Sim':>6}  {'Apps':>4}  {'Age':>4}")
    print(f"  {'─'*3}  {'─'*34}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*4}")
    for r in similar:
        age_str = f"{r['age']:>4.1f}" if r.get("age") is not None else "  - "
        print(
            f"  {r['rank']:>3}  {r['player']:<34}  {r['position']:>4}"
            f"  {r['similarity']:.3f}  {r['appearances']:>4}  {age_str}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find action-style-similar players via TabEventEncoder embeddings"
    )
    p.add_argument(
        "--queries", nargs="+", default=None,
        help="Player name(s) to query. Overrides --demo.",
    )
    p.add_argument(
        "--demo", choices=["elite", "young-stars", "next-gen", "u20", "chelsea", "all"], default="all",
        help="Run built-in demo queries (default: all)",
    )
    p.add_argument("--top-k", type=int, default=TOP_K,
                   help=f"Number of similar players to show (default {TOP_K})")
    p.add_argument(
        "--min-gallery-appearances", type=int, default=MIN_GALLERY_APPEARANCES,
        help=f"Min appearances to enter the gallery (default {MIN_GALLERY_APPEARANCES})",
    )
    p.add_argument(
        "--include-test", action="store_true",
        help="Include test-set matches (Euro 2020 / WC 2022 / Copa 2024) in gallery",
    )
    p.add_argument(
        "--max-gallery-age", type=float, default=None,
        help="If set, restrict gallery candidates to players with age <= this value "
             "at AGE_REF_DATE (2016-06-01). Common values: 21 for U21, 23 for U23.",
    )
    p.add_argument(
        "--age-ref-date", default=AGE_REF_DATE,
        help=f"Reference date for age computation (default {AGE_REF_DATE}).",
    )
    p.add_argument(
        "--birthdate-table", default=BIRTHDATE_TABLE,
        help=f"Path to player birthdate CSV (default {BIRTHDATE_TABLE}).",
    )
    p.add_argument("--max-matches", type=int, default=None,
                   help="Cap match files (useful for quick --demo tests)")
    p.add_argument("--embed-batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore existing cache and rebuild gallery")
    p.add_argument("--output", default=OUTPUT_CSV,
                   help=f"CSV output path (default: {OUTPUT_CSV})")
    p.add_argument("--pretrain-checkpoint", default=PRETRAIN_CHECKPOINT)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"DEVICE = {device}")

    # Load encoder
    print(f"Loading encoder from {args.pretrain_checkpoint} ...")
    encoder, cat_cols, cont_cols, cat_maps, _, _ = load_pretrained_encoder(
        args.pretrain_checkpoint
    )

    # Collect match files
    train_files = list_event_files(TRAIN_ROOT, max_match_files=args.max_matches)
    files = list(train_files)
    if args.include_test:
        test_files = list_event_files(TEST_ROOT, max_match_files=args.max_matches)
        files += list(test_files)
    src_note = "train + test" if args.include_test else "train (2015/16 leagues)"
    print(f"Match files: {len(files)}  [{src_note}]")

    # Build / load player gallery
    names, positions, n_apps, emb_matrix, raw_matrix = build_player_gallery(
        encoder, cat_cols, cont_cols, cat_maps,
        files, device, args.embed_batch_size,
        ckpt_path=args.pretrain_checkpoint,
        use_cache=not args.no_cache,
    )
    eligible = int((n_apps >= args.min_gallery_appearances).sum())
    print(
        f"Gallery: {len(names)} players total  "
        f"(eligible with >= {args.min_gallery_appearances} apps: {eligible})"
    )

    # Optional age filter for gallery candidates.
    ages = _build_age_array(names, args.birthdate_table, args.age_ref_date)
    extra_eligible: Optional[np.ndarray] = None
    if args.max_gallery_age is not None:
        extra_eligible = (~np.isnan(ages)) & (ages <= float(args.max_gallery_age))
        n_age_ok = int(extra_eligible.sum())
        print(
            f"Age filter: candidates with age <= {args.max_gallery_age} "
            f"at {args.age_ref_date} = {n_age_ok}"
        )

    # Determine query list
    if args.queries:
        query_names  = args.queries
        show_context = False
    else:
        demo = args.demo
        if demo == "all":
            query_names = DEMO_QUERIES["elite"] + DEMO_QUERIES["young-stars"] + DEMO_QUERIES["next-gen"] + DEMO_QUERIES["u20"] + DEMO_QUERIES["chelsea"]
        else:
            query_names = DEMO_QUERIES[demo]
        show_context = True

    # Run queries
    all_rows:   List[Dict] = []
    not_found:  List[str]  = []

    for q in query_names:
        idx = resolve_query(q, names)
        if idx is None:
            not_found.append(q)
            continue

        full_name  = str(names[idx])
        pos_label  = str(positions[idx])
        app_count  = int(n_apps[idx])
        ctx_note   = PLAYER_CONTEXT.get(full_name.lower(), "") if show_context else ""

        similar = find_similar(
            idx, names, positions, n_apps, emb_matrix,
            top_k=args.top_k,
            min_gallery_appearances=args.min_gallery_appearances,
            extra_eligible=extra_eligible,
            ages=ages,
        )

        print_results(full_name, pos_label, app_count, similar, ctx_note)

        for r in similar:
            all_rows.append({
                "query":             full_name,
                "query_position":    pos_label,
                "query_appearances": app_count,
                **r,
            })

    # Not-found report
    if not_found:
        print(f"\nNot found in gallery: {not_found}")
        print(
            "  Tip: use partial names e.g. 'Messi', 'De Bruyne', 'Rashford'.\n"
            "  Run with --max-matches to limit files first, then check available names."
        )

    # Save CSV
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nResults saved → {args.output}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
