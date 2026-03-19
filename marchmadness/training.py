"""
training.py - Parameter training against historical tournament outcomes.

Pipeline:
    1. build_training_dataset()  - join Kaggle results with KenPom season data
       and schedule-derived features into one row-per-game frame
    2. train_params()            - minimize log-loss over ModelParams weights
    3. cross_validate()          - leave-one-year-out evaluation
    4. evaluate_params()         - metrics for any params/year subset
"""

from __future__ import annotations

import warnings
from dataclasses import replace as dataclass_replace
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import TeamFeatures, _apply_conf_adjustment, _compute_sigma
from .model import HISTORICAL_SEED_WIN_RATES, ModelParams, win_probability_expected

# ---------------------------------------------------------------------------
# Known Kaggle -> KenPom name mismatches
# ---------------------------------------------------------------------------

DEFAULT_NAME_MAP: dict[str, str] = {
    "North Carolina": "UNC",
    "Connecticut": "UConn",
    "Miami FL": "Miami (FL)",
    "ETSU": "East Tennessee St.",
    "F Dickinson": "Fairleigh Dickinson",
    "FGCU": "Florida Gulf Coast",
    "Nebraska Omaha": "Omaha",
    "Cal St Fullerton": "CS Fullerton",
    "Cal St Northridge": "CSUN",
    "Kent": "Kent St.",
    "Louisiana Lafayette": "Louisiana",
    "Louisiana Monroe": "UL Monroe",
    "Mississippi": "Ole Miss",
    "MTSU": "Middle Tennessee",
    "NC A&T": "North Carolina A&T",
    "NC Central": "North Carolina Central",
    "Loyola Chicago": "Loyola-Chicago",
    "SF Austin": "Stephen F. Austin",
    "St John's": "St. John's",
    "Saint Joseph's": "St. Joseph's",
    "Saint Mary's": "Saint Mary's (CA)",
    "SUNY Albany": "Albany",
    "TAM C. Christi": "Texas A&M Corpus Chris",
    "Abilene Christian": "Abilene Chr.",
    "Tennessee Martin": "UT Martin",
    "Texas A&M Corpus Chris": "A&M-Corpus Christi",
    "UT San Antonio": "UTSA",
    "WKU": "Western Kentucky",
    "Arkansas Little Rock": "Little Rock",
    "UC Santa Barbara": "UCSB",
    "Long Island University": "LIU",
    "Norfolk State": "Norfolk St.",
    "Morehead State": "Morehead St.",
    "Wichita State": "Wichita St.",
    "Weber State": "Weber St.",
    "Utah State": "Utah St.",
    "Colorado State": "Colorado St.",
    "Iowa State": "Iowa St.",
    "Michigan State": "Michigan St.",
    "Ohio State": "Ohio St.",
    "Penn State": "Penn St.",
    "Arizona State": "Arizona St.",
    "Oregon State": "Oregon St.",
    "Washington State": "Washington St.",
    "Kansas State": "Kansas St.",
    "Oklahoma State": "Oklahoma St.",
    "Mississippi State": "Mississippi St.",
    "Arkansas State": "Arkansas St.",
    "Georgia State": "Georgia St.",
    "Boise State": "Boise St.",
    "San Diego State": "San Diego St.",
    "Fresno State": "Fresno St.",
    "Sacramento State": "Sacramento St.",
    "Portland State": "Portland St.",
    "Montana State": "Montana St.",
    "Murray State": "Murray St.",
    "Youngstown State": "Youngstown St.",
    "Southeast Missouri St": "SE Missouri St.",
    "Boston University": "Boston U.",
    "Brigham Young": "BYU",
    "Southern Methodist": "SMU",
    "Texas Christian": "TCU",
}


# ---------------------------------------------------------------------------
# Name matching utilities
# ---------------------------------------------------------------------------

def _fuzzy_match(name: str, candidates: list[str], threshold: float = 0.75) -> Optional[str]:
    """Return the best fuzzy match for `name`, or None if below threshold."""

    best_score = 0.0
    best_match = None
    for candidate in candidates:
        score = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score >= threshold else None


def resolve_team_names(
    kaggle_names: list[str],
    kenpom_names: list[str],
    name_map: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Resolve Kaggle team names to KenPom names.

    Resolution order:
        1. user `name_map`
        2. DEFAULT_NAME_MAP
        3. exact match
        4. fuzzy match
    """

    combined_map = {**DEFAULT_NAME_MAP, **(name_map or {})}
    kenpom_set = set(kenpom_names)

    resolved: dict[str, str] = {}
    unresolved: list[str] = []

    for kaggle_name in kaggle_names:
        if kaggle_name in combined_map and combined_map[kaggle_name] in kenpom_set:
            resolved[kaggle_name] = combined_map[kaggle_name]
            continue

        if kaggle_name in kenpom_set:
            resolved[kaggle_name] = kaggle_name
            continue

        fuzzy = _fuzzy_match(kaggle_name, kenpom_names)
        if fuzzy is not None:
            resolved[kaggle_name] = fuzzy
            continue

        unresolved.append(kaggle_name)

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

def build_training_dataset(
    kaggle_df: pd.DataFrame,
    kenpom_stats: dict[int, pd.DataFrame],
    kaggle_teams_path: Optional[str] = None,
    name_map: Optional[dict[str, str]] = None,
    schedule_features: Optional[dict[int, dict[str, tuple[float, float]]]] = None,
    conf_tourney_results: Optional[dict[int, dict[str, int]]] = None,
    kaggle_features_df: Optional[pd.DataFrame] = None,
    min_year: int = 2010,
    max_year: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Join Kaggle tournament results with KenPom team stats into a flat training frame.

    Args:
        kaggle_df: Output of load_kaggle_historical().
        kenpom_stats: {year: DataFrame} from load_kenpom_historical().
        kaggle_teams_path: Path to Kaggle MTeams.csv.
        name_map: Extra Kaggle -> KenPom overrides.
        schedule_features: {year: {team: (recent_form, season_trajectory)}}.
        conf_tourney_results: {year: {team: conference_tournament_wins}}.
        kaggle_features_df: Output of build_kaggle_features_historical() —
            DataFrame indexed by (Season, TeamID) with neutral court stats,
            coach experience, consensus rankings, etc. When provided, columns
            neutral_win_pct and coach_tourney_wins are added to each training row.
    """

    rng = np.random.default_rng(random_state)

    if kaggle_teams_path is not None:
        teams_df = pd.read_csv(kaggle_teams_path)
        id_to_name = teams_df.set_index("TeamID")["TeamName"].to_dict()
        kaggle_df = kaggle_df.copy()
        kaggle_df["WTeamName"] = kaggle_df["WTeamID"].map(id_to_name)
        kaggle_df["LTeamName"] = kaggle_df["LTeamID"].map(id_to_name)
    elif "WTeamName" not in kaggle_df.columns or "LTeamName" not in kaggle_df.columns:
        raise ValueError(
            "kaggle_df must include WTeamName/LTeamName, or provide kaggle_teams_path."
        )

    df = kaggle_df[kaggle_df["Season"] >= min_year].copy()
    if max_year is not None:
        df = df[df["Season"] <= max_year]

    available_years = sorted(kenpom_stats.keys())
    df = df[df["Season"].isin(available_years)]
    if df.empty:
        raise ValueError(
            "No games remain after filtering.\n"
            f"Kaggle seasons: {sorted(kaggle_df['Season'].unique())}\n"
            f"KenPom seasons: {available_years}"
        )

    all_kenpom_names: list[str] = []
    for year_df in kenpom_stats.values():
        all_kenpom_names.extend(year_df["Team"].tolist())
    all_kenpom_names = list(dict.fromkeys(all_kenpom_names))

    all_kaggle_names = list(set(df["WTeamName"].dropna()) | set(df["LTeamName"].dropna()))
    resolved_names, unresolved = resolve_team_names(all_kaggle_names, all_kenpom_names, name_map)
    if unresolved:
        warnings.warn(
            f"{len(unresolved)} team name(s) could not be matched to KenPom data and will be excluded:\n"
            f"  {sorted(unresolved)}",
            stacklevel=2,
        )

    # Build KenPom name -> Kaggle TeamID for Kaggle feature lookup,
    # then flatten kaggle_features_df to a plain dict so we avoid pandas
    # MultiIndex ambiguity (loc can return a DataFrame instead of a Series
    # depending on whether the index is sorted/unique).
    kenpom_to_kaggle_id: dict[str, int] = {}
    kaggle_lookup: dict[tuple, dict] = {}   # {(season, team_id): {col: val}}
    if kaggle_features_df is not None and kaggle_teams_path is not None:
        # id_to_name: TeamID -> Kaggle team name (from MTeams.csv)
        # resolved_names: Kaggle team name -> KenPom name
        # Build reverse: KenPom name -> TeamID
        for team_id, kaggle_name in id_to_name.items():
            kenpom_name = resolved_names.get(kaggle_name)
            if kenpom_name:
                kenpom_to_kaggle_id[kenpom_name] = int(team_id)

        # Flatten to plain dict — avoids all MultiIndex .loc ambiguity
        for (season, team_id), row in kaggle_features_df.iterrows():
            kaggle_lookup[(int(season), int(team_id))] = row.to_dict()

    year_indexes: dict[int, pd.DataFrame] = {}
    for year, year_df in kenpom_stats.items():
        year_indexes[year] = _apply_conf_adjustment(year_df).set_index("Team")

    rows: list[dict[str, object]] = []
    skipped = 0
    schedule_slots_found = 0
    total_slots = 0

    for _, game in df.iterrows():
        year = int(game["Season"])
        w_name = game["WTeamName"]
        l_name = game["LTeamName"]

        if pd.isna(w_name) or pd.isna(l_name):
            skipped += 1
            continue

        kenpom_w = resolved_names.get(w_name)
        kenpom_l = resolved_names.get(l_name)
        if kenpom_w is None or kenpom_l is None:
            skipped += 1
            continue

        index_df = year_indexes.get(year)
        if index_df is None or kenpom_w not in index_df.index or kenpom_l not in index_df.index:
            skipped += 1
            continue

        row_w = index_df.loc[kenpom_w]
        row_l = index_df.loc[kenpom_l]
        w_seed = int(game["WSeed"])
        l_seed = int(game["LSeed"])

        if rng.random() < 0.5:
            team_a, team_b = kenpom_w, kenpom_l
            row_a, row_b = row_w, row_l
            seed_a, seed_b = w_seed, l_seed
            label = 1
        else:
            team_a, team_b = kenpom_l, kenpom_w
            row_a, row_b = row_l, row_w
            seed_a, seed_b = l_seed, w_seed
            label = 0

        year_schedule = schedule_features.get(year, {}) if schedule_features else {}
        year_conf = conf_tourney_results.get(year, {}) if conf_tourney_results else {}

        has_schedule_a = team_a in year_schedule
        has_schedule_b = team_b in year_schedule
        schedule_slots_found += int(has_schedule_a) + int(has_schedule_b)
        total_slots += 2

        recent_form_a, season_trajectory_a = year_schedule.get(team_a, (0.0, 0.0))
        recent_form_b, season_trajectory_b = year_schedule.get(team_b, (0.0, 0.0))
        conf_tourney_a = int(year_conf.get(team_a, 0))
        conf_tourney_b = int(year_conf.get(team_b, 0))

        # Kaggle-derived features (neutral court win%, coach experience)
        neutral_win_pct_a = 0.5
        neutral_win_pct_b = 0.5
        coach_wins_a = 0
        coach_wins_b = 0
        if kaggle_lookup:
            for team_name, is_a in [(team_a, True), (team_b, False)]:
                kid = kenpom_to_kaggle_id.get(team_name)
                if kid is not None:
                    kdata = kaggle_lookup.get((year, kid))
                    if kdata is not None:
                        nwp = kdata.get("neutral_win_pct", np.nan)
                        cwins = kdata.get("coach_tourney_wins", np.nan)
                        if is_a:
                            if pd.notna(nwp):
                                neutral_win_pct_a = float(nwp)
                            if pd.notna(cwins):
                                coach_wins_a = int(cwins)
                        else:
                            if pd.notna(nwp):
                                neutral_win_pct_b = float(nwp)
                            if pd.notna(cwins):
                                coach_wins_b = int(cwins)

        def _get(row: pd.Series, column: str, default: float = 0.0) -> float:
            return float(row[column]) if column in row.index and pd.notna(row[column]) else default

        win_pct_a = _get(row_a, "WinPct", 0.5)
        win_pct_b = _get(row_b, "WinPct", 0.5)

        row_dict: dict[str, object] = {
            "year": year,
            "team_a": team_a,
            "team_b": team_b,
            "label": label,
            "day_num": int(game["DayNum"]) if "DayNum" in game.index else 0,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "adj_o_a": _get(row_a, "AdjO"),
            "adj_d_a": _get(row_a, "AdjD"),
            "adj_t_a": _get(row_a, "AdjT", 68.0),
            "luck_a": _get(row_a, "Luck"),
            "recent_form_a": float(recent_form_a),
            "season_trajectory_a": float(season_trajectory_a),
            "conf_tourney_wins_a": conf_tourney_a,
            "win_pct_a": win_pct_a,
            "sigma_a": _compute_sigma(win_pct_a),
            "neutral_win_pct_a": neutral_win_pct_a,
            "coach_tourney_wins_a": coach_wins_a,
            "adj_o_b": _get(row_b, "AdjO"),
            "adj_d_b": _get(row_b, "AdjD"),
            "adj_t_b": _get(row_b, "AdjT", 68.0),
            "luck_b": _get(row_b, "Luck"),
            "recent_form_b": float(recent_form_b),
            "season_trajectory_b": float(season_trajectory_b),
            "conf_tourney_wins_b": conf_tourney_b,
            "win_pct_b": win_pct_b,
            "sigma_b": _compute_sigma(win_pct_b),
            "neutral_win_pct_b": neutral_win_pct_b,
            "coach_tourney_wins_b": coach_wins_b,
        }

        for col, suffix in [
            ("EFG_O", "efg_o"),
            ("EFG_D", "efg_d"),
            ("TOV_O", "tov_o"),
            ("TOV_D", "tov_d"),
            ("ORB", "orb"),
            ("DRB", "drb"),
            ("FTR_O", "ftr_o"),
            ("FTR_D", "ftr_d"),
            ("AdjEM", "adj_em"),
        ]:
            row_dict[f"{suffix}_a"] = _get(row_a, col)
            row_dict[f"{suffix}_b"] = _get(row_b, col)

        rows.append(row_dict)

    if skipped:
        warnings.warn(
            f"{skipped} games skipped due to missing team stats "
            f"({skipped / (len(rows) + skipped):.1%} of total).",
            stacklevel=2,
        )

    result = pd.DataFrame(rows)
    coverage = (schedule_slots_found / total_slots * 100.0) if total_slots else 0.0
    print(
        f"Training dataset: {len(result)} games | "
        f"{result['year'].nunique()} seasons "
        f"({result['year'].min()}-{result['year'].max()}) | "
        f"schedule coverage: {coverage:.0f}% of team-slots"
    )
    return result


# ---------------------------------------------------------------------------
# Feature scaler — standardize secondary linear features
# ---------------------------------------------------------------------------

# Features to standardize. adj_o/adj_d/adj_t are excluded because they enter
# the model via a non-linear ratio formula; standardizing them would break it.
SCALED_FEATURES = (
    "luck", "recent_form", "season_trajectory",
    "neutral_win_pct", "coach_tourney_wins",
    # Four factors — standardized so matchup differences are on a common scale
    "efg_o", "efg_d", "tov_o", "tov_d", "orb", "drb", "ftr_o", "ftr_d",
)


def fit_feature_scaler(training_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Fit a zero-mean / unit-variance scaler on the training set secondary features.

    Pools the _a and _b columns for each feature so the scaler is fitted on the
    full distribution of tournament-team values (not just one side's assignment).

    Returns a dict: {feature_name: {"mean": float, "std": float}}

    Example:
        scaler = fit_feature_scaler(training_df)
        # → {"luck": {"mean": 0.12, "std": 0.74},
        #    "recent_form": {"mean": 0.44, "std": 0.18},
        #    "season_trajectory": {"mean": 0.0003, "std": 0.0028}}
    """
    scaler: dict[str, dict[str, float]] = {}
    for feat in SCALED_FEATURES:
        col_a = f"{feat}_a"
        col_b = f"{feat}_b"
        vals = pd.concat([
            training_df[col_a] if col_a in training_df.columns
            else pd.Series(dtype=float),
            training_df[col_b] if col_b in training_df.columns
            else pd.Series(dtype=float),
        ]).dropna()

        mean = float(vals.mean()) if len(vals) else 0.0
        std  = float(vals.std())  if len(vals) > 1 else 1.0
        if std < 1e-9:
            std = 1.0
        scaler[feat] = {"mean": mean, "std": std}

    return scaler


def apply_feature_scaler(
    df: pd.DataFrame,
    scaler: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Apply a fitted scaler to a training DataFrame in-place (returns copy).

    Standardizes the _a and _b columns for every feature in `scaler`.
    Call this on the training DataFrame before passing it to train_params().
    """
    df = df.copy()
    for feat, stats in scaler.items():
        mean, std = stats["mean"], stats["std"]
        for side in ("a", "b"):
            col = f"{feat}_{side}"
            if col in df.columns:
                df[col] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# Feature conversion and vectorized probability path
# ---------------------------------------------------------------------------

def _row_to_features(row, side: str) -> TeamFeatures:
    """Build a TeamFeatures object from one training row side."""

    def _f(attr: str, default: float = 0.0) -> float:
        return float(getattr(row, attr, default))

    return TeamFeatures(
        name=getattr(row, f"team_{side}"),
        adj_o=_f(f"adj_o_{side}"),
        adj_d=_f(f"adj_d_{side}"),
        adj_t=_f(f"adj_t_{side}", 68.0),
        luck=_f(f"luck_{side}"),
        win_pct=_f(f"win_pct_{side}", 0.5),
        conf="",
        seed=int(getattr(row, f"seed_{side}")),
        region=None,
        recent_form=_f(f"recent_form_{side}"),
        season_trajectory=_f(f"season_trajectory_{side}"),
        conf_tourney_wins=int(_f(f"conf_tourney_wins_{side}")),
        sigma=_f(f"sigma_{side}", 0.3),
        efg_o=_f(f"efg_o_{side}"),
        efg_d=_f(f"efg_d_{side}"),
        tov_o=_f(f"tov_o_{side}"),
        tov_d=_f(f"tov_d_{side}"),
        orb=_f(f"orb_{side}"),
        drb=_f(f"drb_{side}"),
        ftr_o=_f(f"ftr_o_{side}"),
        ftr_d=_f(f"ftr_d_{side}"),
        neutral_win_pct=_f(f"neutral_win_pct_{side}", 0.5),
        tourney_coach_wins=int(_f(f"coach_tourney_wins_{side}")),
    )


def _seed_prior_scalar(seed_a: int, seed_b: int, weight: float) -> float:
    if seed_a == seed_b:
        return 0.0
    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
    rate = HISTORICAL_SEED_WIN_RATES.get((lo, hi))
    if rate is None:
        rate = min(0.5 + (hi - lo) * 0.04, 0.99)
    logit_val = float(np.clip(np.log(rate / (1 - rate)), -2.0, 2.0))
    if seed_a > seed_b:
        logit_val = -logit_val
    return weight * logit_val


def _expected_probabilities(params: ModelParams, training_df: pd.DataFrame) -> np.ndarray:
    """Vectorized expected win probabilities for a training frame."""

    p_ref = params.league_avg_adj_o

    adj_o_a = training_df["adj_o_a"].to_numpy(dtype=float)
    adj_o_b = training_df["adj_o_b"].to_numpy(dtype=float)
    adj_d_a = training_df["adj_d_a"].to_numpy(dtype=float)
    adj_d_b = training_df["adj_d_b"].to_numpy(dtype=float)
    adj_t_a = training_df["adj_t_a"].to_numpy(dtype=float)
    adj_t_b = training_df["adj_t_b"].to_numpy(dtype=float)
    luck_a = training_df["luck_a"].to_numpy(dtype=float)
    luck_b = training_df["luck_b"].to_numpy(dtype=float)
    recent_form_a = training_df.get("recent_form_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    recent_form_b = training_df.get("recent_form_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    trajectory_a = training_df.get("season_trajectory_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    trajectory_b = training_df.get("season_trajectory_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    conf_wins_a = training_df.get("conf_tourney_wins_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    conf_wins_b = training_df.get("conf_tourney_wins_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    neutral_wp_a = training_df.get("neutral_win_pct_a", pd.Series(0.5, index=training_df.index)).to_numpy(dtype=float)
    neutral_wp_b = training_df.get("neutral_win_pct_b", pd.Series(0.5, index=training_df.index)).to_numpy(dtype=float)
    coach_wins_a = training_df.get("coach_tourney_wins_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    coach_wins_b = training_df.get("coach_tourney_wins_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    # Four factors (standardized)
    efg_o_a = training_df.get("efg_o_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    efg_o_b = training_df.get("efg_o_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    efg_d_a = training_df.get("efg_d_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    efg_d_b = training_df.get("efg_d_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    tov_o_a = training_df.get("tov_o_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    tov_o_b = training_df.get("tov_o_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    tov_d_a = training_df.get("tov_d_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    tov_d_b = training_df.get("tov_d_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    orb_a   = training_df.get("orb_a",   pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    orb_b   = training_df.get("orb_b",   pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    drb_a   = training_df.get("drb_a",   pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    drb_b   = training_df.get("drb_b",   pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    ftr_o_a = training_df.get("ftr_o_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    ftr_o_b = training_df.get("ftr_o_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    ftr_d_a = training_df.get("ftr_d_a", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    ftr_d_b = training_df.get("ftr_d_b", pd.Series(0.0, index=training_df.index)).to_numpy(dtype=float)
    sigma_a = training_df["sigma_a"].to_numpy(dtype=float)
    sigma_b = training_df["sigma_b"].to_numpy(dtype=float)
    seed_a = training_df["seed_a"].to_numpy(dtype=int)
    seed_b = training_df["seed_b"].to_numpy(dtype=int)

    a_pts = adj_o_a * (adj_d_b / p_ref)
    b_pts = adj_o_b * (adj_d_a / p_ref)
    raw_margin = a_pts - b_pts
    game_pace = (adj_t_a + adj_t_b) / 2.0
    base_margin = raw_margin * (game_pace / params.league_avg_adj_t) / params.margin_scale

    recent_form_contrib = params.recent_form_weight * (recent_form_a - recent_form_b)
    trajectory_contrib = params.trajectory_weight * (trajectory_a - trajectory_b)
    luck_contrib = params.luck_weight * (luck_a - luck_b)
    conf_contrib = params.conf_tourney_weight * (conf_wins_a - conf_wins_b)
    neutral_contrib = params.neutral_win_pct_weight * (neutral_wp_a - neutral_wp_b)
    coach_contrib = params.coach_exp_weight * (coach_wins_a - coach_wins_b)
    # Four-factor matchup: each term = A's advantage over B in that dimension
    efg_net = (efg_o_a - efg_d_b) - (efg_o_b - efg_d_a)
    to_net  = (tov_d_b - tov_o_a) - (tov_d_a - tov_o_b)
    orb_net = (orb_a   - drb_b)   - (orb_b   - drb_a)
    ftr_net = (ftr_o_a - ftr_d_b) - (ftr_o_b - ftr_d_a)
    ff_contrib = (
        params.w_efg * efg_net
        + params.w_to  * to_net
        + params.w_orb * orb_net
        + params.w_ftr * ftr_net
    )
    seed_contrib = np.array(
        [_seed_prior_scalar(int(sa), int(sb), params.seed_prior_weight) for sa, sb in zip(seed_a, seed_b)],
        dtype=float,
    )

    mu = base_margin + recent_form_contrib + trajectory_contrib + luck_contrib + conf_contrib + neutral_contrib + coach_contrib + ff_contrib + seed_contrib

    sigma_team = np.clip((sigma_a + sigma_b) / 2.0, 0.1, 0.5)
    safe_df = max(float(params.shock_df), 3.0)
    shock_var = 2.0 * (params.shock_scale ** 2) * safe_df / (safe_df - 2.0)
    tau = max(float(params.temperature), 0.1)
    sigma_eff = np.sqrt(sigma_team ** 2 + shock_var) * tau

    probs = 1.0 / (1.0 + np.exp(-mu / sigma_eff))
    return np.clip(probs, 1e-7, 1 - 1e-7)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _log_loss(params: ModelParams, training_df: pd.DataFrame) -> float:
    """Mean log-loss over all games in `training_df`."""

    eps = 1e-7
    total = 0.0
    for row in training_df.itertuples(index=False):
        ft_a = _row_to_features(row, "a")
        ft_b = _row_to_features(row, "b")
        p = float(np.clip(win_probability_expected(ft_a, ft_b, params), eps, 1 - eps))
        y = float(row.label)
        total += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / len(training_df)


def _log_loss_vectorized(params: ModelParams, training_df: pd.DataFrame) -> float:
    """Vectorized log-loss computation used inside the optimizer."""

    probs = _expected_probabilities(params, training_df)
    labels = training_df["label"].to_numpy(dtype=float)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(loss)


# ---------------------------------------------------------------------------
# Parameter training
# ---------------------------------------------------------------------------

# Trained weights. Bounds allow negative for four-factor weights because after
# standardization the sign of each factor's contribution isn't guaranteed.
# trajectory/seed_prior/neutral_win_pct/coach_exp zero out in CV and are excluded.
_PARAM_NAMES = [
    "luck_weight",
    "recent_form_weight",
    "seed_prior_weight",
    "w_efg",
    "w_to",
    "w_orb",
    "w_ftr",
]
_PARAM_BOUNDS = [
    (0.0, 20.0),    # luck_weight
    (0.0, 20.0),    # recent_form_weight
    (0.0, 2.0),     # seed_prior_weight
    (-20.0, 20.0),  # w_efg
    (-20.0, 20.0),  # w_to
    (-20.0, 20.0),  # w_orb
    (-20.0, 20.0),  # w_ftr
]


def _pack(params: ModelParams) -> np.ndarray:
    return np.array(
        [
            params.luck_weight,
            params.recent_form_weight,
            params.seed_prior_weight,
            params.w_efg,
            params.w_to,
            params.w_orb,
            params.w_ftr,
        ],
        dtype=float,
    )


def _unpack(x: np.ndarray, base_params: ModelParams) -> ModelParams:
    return dataclass_replace(
        base_params,
        luck_weight=float(x[0]),
        recent_form_weight=float(x[1]),
        seed_prior_weight=float(x[2]),
        w_efg=float(x[3]),
        w_to=float(x[4]),
        w_orb=float(x[5]),
        w_ftr=float(x[6]),
    )


def train_params(
    training_df: pd.DataFrame,
    params_init: Optional[ModelParams] = None,
    train_years: Optional[list[int]] = None,
    verbose: bool = True,
) -> ModelParams:
    """
    Optimize feature weights to minimize log-loss on historical NCAA tournament games.

    shock_df and shock_scale must be pre-calibrated via calibrate_shock_params()
    and set on params_init before calling this function.  Only the four secondary
    feature weights are optimised here: luck_weight, recent_form_weight,
    trajectory_weight, seed_prior_weight.

    Rationale: shock params control "how upsetty is March Madness in general"
    and are best calibrated against historical seed-matchup upset rates.  Feature
    weights control "given two specific teams, how much does each signal matter"
    and are best fitted by log-loss.  Mixing both into one objective causes the
    optimizer to eliminate shock noise entirely (log-loss always prefers certainty).
    """

    if params_init is None:
        params_init = ModelParams()

    df = training_df.copy()
    if train_years is not None:
        df = df[df["year"].isin(train_years)]
    if df.empty:
        raise ValueError("No training data after year filter.")

    x0 = _pack(params_init)

    def objective(x: np.ndarray) -> float:
        return _log_loss_vectorized(_unpack(x, params_init), df)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=_PARAM_BOUNDS,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    best_params = _unpack(result.x, params_init)

    if verbose:
        print(f"log-loss: {result.fun:.6f}  (random baseline = 0.6931)")
        print(
            f"shock_df={best_params.shock_df:.2f}  shock_scale={best_params.shock_scale:.4f}  "
            f"luck_w={best_params.luck_weight:.4f}  recent_form_w={best_params.recent_form_weight:.4f}  "
            f"w_efg={best_params.w_efg:.4f}  w_to={best_params.w_to:.4f}  "
            f"w_orb={best_params.w_orb:.4f}  w_ftr={best_params.w_ftr:.4f}"
        )

    return best_params


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_params(
    training_df: pd.DataFrame,
    params: ModelParams,
    years: Optional[list[int]] = None,
) -> dict[str, object]:
    """Compute probabilistic metrics, baseline accuracies, and upset-rate diagnostics."""

    df = training_df.copy()
    if years is not None:
        df = df[df["year"].isin(years)]

    if df.empty:
        return {
            "log_loss": np.nan,
            "brier_score": np.nan,
            "accuracy": np.nan,
            "seed_baseline_accuracy": np.nan,
            "adj_em_baseline_accuracy": np.nan,
            "mean_confidence": np.nan,
            "high_confidence_accuracy": np.nan,
            "high_confidence_share": np.nan,
            "n_games": 0,
            "upset_rates": {},
        }

    probs = _expected_probabilities(params, df)
    labels = df["label"].to_numpy(dtype=float)
    seed_a = df["seed_a"].to_numpy(dtype=int)
    seed_b = df["seed_b"].to_numpy(dtype=int)
    adj_em_a = (df["adj_o_a"] - df["adj_d_a"]).to_numpy(dtype=float)
    adj_em_b = (df["adj_o_b"] - df["adj_d_b"]).to_numpy(dtype=float)

    log_loss = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))
    brier_score = float(np.mean((probs - labels) ** 2))
    accuracy = float(np.mean((probs >= 0.5) == labels))

    seed_pred = np.where(
        seed_a < seed_b,
        1.0,
        np.where(seed_a > seed_b, 0.0, (adj_em_a >= adj_em_b).astype(float)),
    )
    seed_baseline_accuracy = float(np.mean(seed_pred == labels))

    adj_em_pred = (adj_em_a >= adj_em_b).astype(float)
    adj_em_baseline_accuracy = float(np.mean(adj_em_pred == labels))

    confidence = np.maximum(probs, 1.0 - probs)
    mean_confidence = float(np.mean(confidence))
    high_confidence_mask = confidence >= 0.70
    high_confidence_share = float(np.mean(high_confidence_mask))
    if high_confidence_mask.any():
        high_confidence_accuracy = float(
            np.mean((probs[high_confidence_mask] >= 0.5) == labels[high_confidence_mask])
        )
    else:
        high_confidence_accuracy = np.nan

    upset_rates: dict[tuple[int, int], float] = {}
    for lo, hi in HISTORICAL_SEED_WIN_RATES:
        mask = (
            ((df["seed_a"] == lo) & (df["seed_b"] == hi))
            | ((df["seed_a"] == hi) & (df["seed_b"] == lo))
        )
        if not mask.any():
            continue

        probs_sub = probs[mask.to_numpy()]
        labels_sub = labels[mask.to_numpy()]
        flipped = (df.loc[mask, "seed_a"] > df.loc[mask, "seed_b"]).to_numpy()
        probs_sub = np.where(flipped, 1.0 - probs_sub, probs_sub)
        labels_sub = np.where(flipped, 1.0 - labels_sub, labels_sub)
        upset_rates[(lo, hi)] = float(np.mean(1.0 - labels_sub))

    return {
        "log_loss": log_loss,
        "brier_score": brier_score,
        "accuracy": accuracy,
        "seed_baseline_accuracy": seed_baseline_accuracy,
        "adj_em_baseline_accuracy": adj_em_baseline_accuracy,
        "mean_confidence": mean_confidence,
        "high_confidence_accuracy": high_confidence_accuracy,
        "high_confidence_share": high_confidence_share,
        "n_games": len(df),
        "upset_rates": upset_rates,
    }


def predict_probs(params: ModelParams, training_df: pd.DataFrame) -> np.ndarray:
    """Return win-probability for team_a for every row in ``training_df``.

    Thin public wrapper around the private vectorised implementation used by
    the optimizer and ``evaluate_params``.
    """
    return _expected_probabilities(params, training_df)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    training_df: pd.DataFrame,
    params_init: Optional[ModelParams] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-one-year-out cross-validation.

    shock_df and shock_scale must be pre-calibrated on params_init before
    calling — they are held fixed across all folds (same rationale as train_params).
    """

    years = sorted(training_df["year"].unique())
    if len(years) < 3:
        raise ValueError(f"Need at least 3 years for cross-validation, got {len(years)}.")

    records: list[dict[str, object]] = []

    for val_year in years:
        train_years = [year for year in years if year != val_year]
        train_df = training_df[training_df["year"].isin(train_years)]
        val_df = training_df[training_df["year"] == val_year]

        if verbose:
            print(
                f"\n--- Fold: val={val_year}  train={train_years[0]}-{train_years[-1]} "
                f"({len(train_df)} train, {len(val_df)} val) ---"
            )

        fold_params = train_params(
            train_df,
            params_init=params_init,
            verbose=verbose,
        )
        metrics = evaluate_params(val_df, fold_params)

        records.append(
            {
                "year": val_year,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "accuracy": metrics["accuracy"],
                "shock_scale": fold_params.shock_scale,
                "shock_df": fold_params.shock_df,
                "luck_weight": fold_params.luck_weight,
                "recent_form_weight": fold_params.recent_form_weight,
                "seed_prior_weight": fold_params.seed_prior_weight,
                "w_efg": fold_params.w_efg,
                "w_to": fold_params.w_to,
                "w_orb": fold_params.w_orb,
                "w_ftr": fold_params.w_ftr,
            }
        )

        if verbose:
            latest = records[-1]
            print(
                f"  -> val log_loss={latest['log_loss']:.4f}  "
                f"brier={latest['brier_score']:.4f}  "
                f"acc={latest['accuracy']:.3f}"
            )

    cv_df = pd.DataFrame(records)

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"CV Summary ({len(years)} folds):")
        print(f"  log_loss:    {cv_df['log_loss'].mean():.4f} +- {cv_df['log_loss'].std():.4f}")
        print(f"  brier_score: {cv_df['brier_score'].mean():.4f} +- {cv_df['brier_score'].std():.4f}")
        print(f"  accuracy:    {cv_df['accuracy'].mean():.3f} +- {cv_df['accuracy'].std():.3f}")
        print(f"  luck_weight        (mean): {cv_df['luck_weight'].mean():.4f} +- {cv_df['luck_weight'].std():.4f}")
        print(f"  recent_form_weight (mean): {cv_df['recent_form_weight'].mean():.4f} +- {cv_df['recent_form_weight'].std():.4f}")
        print(f"  seed_prior_weight  (mean): {cv_df['seed_prior_weight'].mean():.4f} +- {cv_df['seed_prior_weight'].std():.4f}")
        print(f"  w_efg              (mean): {cv_df['w_efg'].mean():.4f} +- {cv_df['w_efg'].std():.4f}")
        print(f"  w_to               (mean): {cv_df['w_to'].mean():.4f} +- {cv_df['w_to'].std():.4f}")
        print(f"  w_orb              (mean): {cv_df['w_orb'].mean():.4f} +- {cv_df['w_orb'].std():.4f}")
        print(f"  w_ftr              (mean): {cv_df['w_ftr'].mean():.4f} +- {cv_df['w_ftr'].std():.4f}")

    return cv_df
