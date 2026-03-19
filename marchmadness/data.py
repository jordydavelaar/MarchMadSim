"""
data.py - Data loading and feature engineering.

Provides the TeamFeatures dataclass plus helpers for:
  - loading manual KenPom CSV exports
  - fetching cached KenPom season/schedule data via kenpompy
  - deriving schedule-based features for recent form and trajectory
  - loading Kaggle historical tournament results
"""

from __future__ import annotations

import re
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_KENPOM_COLS = {"Team", "AdjO", "AdjD", "AdjT", "Luck", "W-L", "Conf"}
BASELINE_SIGMA = 0.3
CONF_ADJ_WEIGHT = 0.1
DEFAULT_UNRANKED_OPPONENT = 400
DEFAULT_RECENT_WINDOW = 10
POSSESSION_FTA_WEIGHT = 0.475
LUCK_PYTH_EXPONENT = 11.5
KAGGLE_RATING_ITERATIONS = 24
KAGGLE_RATING_DAMPING = 0.6
KAGGLE_HOME_EFF_MULTIPLIER = 1.014

_SCHEDULE_COLUMNS = [
    "Date",
    "Team Rank",
    "Opponent Rank",
    "Opponent Name",
    "Result",
    "Possession Number",
    "Location",
    "Record",
    "Conference",
    "Postseason",
]
_POSTSEASON_EXCLUDE_PREFIXES = ("NCAA", "NIT", "CBI", "CIT", "VEGAS", "CROWN")
_MONTH_TO_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# ---------------------------------------------------------------------------
# TeamFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class TeamFeatures:
    """All per-team features consumed downstream."""

    name: str
    adj_o: float
    adj_d: float
    adj_t: float
    luck: float
    win_pct: float
    conf: str
    seed: Optional[int]
    region: Optional[str]
    recent_form: float = 0.0
    season_trajectory: float = 0.0
    conf_tourney_wins: int = 0
    sigma: float = BASELINE_SIGMA
    # Four factors (from KenPom adjusted data; 0.0 = unavailable → no contribution)
    efg_o: float = 0.0     # offensive effective FG%
    efg_d: float = 0.0     # defensive eFG% allowed
    tov_o: float = 0.0     # offensive turnover rate (lower = better)
    tov_d: float = 0.0     # defensive turnover rate = forced TOs (higher = better)
    orb: float = 0.0       # offensive rebound rate
    drb: float = 0.0       # defensive rebound rate
    ftr_o: float = 0.0     # offensive free throw rate
    ftr_d: float = 0.0     # defensive FTR allowed
    # Kaggle-derived features (optional; default to neutral values when unavailable)
    neutral_win_pct: float = 0.5      # win % on neutral courts during regular season
    tourney_coach_wins: int = 0       # career NCAA tournament wins for this team's coach


# ---------------------------------------------------------------------------
# KenPom loading
# ---------------------------------------------------------------------------

def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finalize_kenpom_frame(
    df: pd.DataFrame,
    name_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize fetched or manually loaded KenPom data into one schema."""

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(
        columns={
            "Conference": "Conf",
            "AdjOE": "AdjO",
            "AdjDE": "AdjD",
            "AdjTempo": "AdjT",
            "Tempo-Adj": "AdjT",
            "Off. Efficiency-Adj": "AdjO",
            "Def. Efficiency-Adj": "AdjD",
            "TeamName": "Team",
        }
    )

    if name_map:
        df["Team"] = df["Team"].replace(name_map)

    if "W-L" not in df.columns:
        if {"W", "L"}.issubset(df.columns):
            df["W-L"] = df["W"].astype(str) + "-" + df["L"].astype(str)
        elif {"Wins", "Losses"}.issubset(df.columns):
            df["W-L"] = df["Wins"].astype(str) + "-" + df["Losses"].astype(str)
        else:
            df["W-L"] = "0-0"

    if "Luck" not in df.columns:
        df["Luck"] = 0.0

    if "Conf" not in df.columns:
        df["Conf"] = ""

    df = _coerce_numeric(
        df,
        [
            "AdjO",
            "AdjD",
            "AdjT",
            "Luck",
            "AdjEM",
            "EFG_O",
            "EFG_D",
            "TOV_O",
            "TOV_D",
            "ORB",
            "DRB",
            "FTR_O",
            "FTR_D",
        ],
    )

    missing = REQUIRED_KENPOM_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"KenPom data is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    parsed_record = df["W-L"].astype(str).str.extract(r"(\d+)-(\d+)")
    df["Wins"] = pd.to_numeric(parsed_record[0], errors="coerce").fillna(0).astype(int)
    df["Losses"] = pd.to_numeric(parsed_record[1], errors="coerce").fillna(0).astype(int)
    total_games = df["Wins"] + df["Losses"]
    df["WinPct"] = df["Wins"] / total_games.replace(0, 1)

    df = df.dropna(subset=["Team", "AdjO", "AdjD", "AdjT"]).drop_duplicates(subset=["Team"])

    luck_std = float(df["Luck"].std())
    if luck_std > 0:
        df["Luck"] = (df["Luck"] - df["Luck"].mean()) / luck_std
    else:
        df["Luck"] = 0.0

    return df.reset_index(drop=True)


def load_kenpom(path: str, name_map: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Load a manually-downloaded KenPom CSV."""

    return _finalize_kenpom_frame(pd.read_csv(path), name_map=name_map)


def _merge_four_factors(base_df: pd.DataFrame, ff_df: pd.DataFrame) -> pd.DataFrame:
    ff_df = ff_df.copy()
    ff_df.columns = [str(col).strip() for col in ff_df.columns]
    ff_df = ff_df.rename(columns={"Conference": "Conf", "TeamName": "Team"})

    rename_map = {
        "AdjOE": "AdjO",
        "AdjDE": "AdjD",
        "AdjTempo": "AdjT",
        "Off-eFG%": "EFG_O",
        "Def-eFG%": "EFG_D",
        "Off-TO%": "TOV_O",
        "Def-TO%": "TOV_D",
        "Off-OR%": "ORB",
        "Off-FTRate": "FTR_O",
        "Def-FTRate": "FTR_D",
    }
    ff_df = ff_df.rename(columns=rename_map)

    if "Def-OR%" in ff_df.columns and "DRB" not in ff_df.columns:
        ff_df["DRB"] = 100.0 - pd.to_numeric(ff_df["Def-OR%"], errors="coerce")

    keep_cols = ["Team"]
    keep_cols.extend(
        col
        for col in ["AdjO", "AdjD", "AdjT", "EFG_O", "EFG_D", "TOV_O", "TOV_D", "ORB", "DRB", "FTR_O", "FTR_D"]
        if col in ff_df.columns
    )

    if len(keep_cols) == 1:
        return base_df

    merged = base_df.merge(ff_df[keep_cols], on="Team", how="left", suffixes=("", "_ff"))
    for col in ["AdjO", "AdjD", "AdjT"]:
        ff_col = f"{col}_ff"
        if ff_col in merged.columns:
            merged[col] = merged[col].fillna(merged[ff_col])
            merged = merged.drop(columns=[ff_col])
    return merged


def fetch_kenpom_season(
    browser,
    year: int,
    cache_dir: str = "data/kenpom/seasons",
    force_refresh: bool = False,
    name_map: Optional[dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch one KenPom season via kenpompy, or load it from cache.

    Cached files are stored at `{cache_dir}/{year}.csv`.
    """

    cache_path = Path(cache_dir) / f"{year}.csv"
    if cache_path.exists() and not force_refresh:
        return _finalize_kenpom_frame(pd.read_csv(cache_path), name_map=name_map)

    if browser is None:
        warnings.warn(
            f"Missing browser for uncached KenPom season fetch: year={year}.",
            stacklevel=2,
        )
        return None

    try:
        from kenpompy.misc import get_pomeroy_ratings
        from kenpompy.summary import get_efficiency, get_fourfactors

        season_arg = str(year)
        ratings_df = get_pomeroy_ratings(browser, season=season_arg)
        ratings_df = _merge_four_factors(ratings_df, get_fourfactors(browser, season=season_arg))

        if "AdjT" not in ratings_df.columns:
            eff_df = get_efficiency(browser, season=season_arg)
            eff_df = eff_df.rename(
                columns={
                    "TeamName": "Team",
                    "Tempo-Adj": "AdjT",
                    "Off. Efficiency-Adj": "AdjO",
                    "Def. Efficiency-Adj": "AdjD",
                }
            )
            keep_cols = [col for col in ["Team", "AdjO", "AdjD", "AdjT"] if col in eff_df.columns]
            ratings_df = ratings_df.merge(eff_df[keep_cols], on="Team", how="left", suffixes=("", "_eff"))
            for col in ["AdjO", "AdjD", "AdjT"]:
                eff_col = f"{col}_eff"
                if eff_col in ratings_df.columns:
                    ratings_df[col] = ratings_df[col].fillna(ratings_df[eff_col])
                    ratings_df = ratings_df.drop(columns=[eff_col])

        final_df = _finalize_kenpom_frame(ratings_df, name_map=name_map)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(cache_path, index=False)
        return final_df
    except Exception as exc:
        warnings.warn(
            f"Could not fetch KenPom season stats for year={year}: {exc}",
            stacklevel=2,
        )
        return None


def load_kenpom_historical(
    years: list[int],
    browser=None,
    cache_dir: str = "data/kenpom/seasons",
    force_refresh: bool = False,
    name_map: Optional[dict[str, str]] = None,
    prefetch_schedules: bool = True,
    schedule_cache_dir: str = "data/kenpom/schedules",
    schedule_force_refresh: bool = False,
    schedule_progress: bool = True,
) -> dict[int, pd.DataFrame]:
    """
    Load or fetch cached KenPom season data for multiple years.

    By default this also warms the per-team schedule cache so downstream
    training steps can derive schedule-based features without triggering a
    second large download pass.
    """

    result: dict[int, pd.DataFrame] = {}
    for year in years:
        season_df = fetch_kenpom_season(
            browser=browser,
            year=year,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            name_map=name_map,
        )
        if season_df is not None:
            result[year] = season_df
            if prefetch_schedules:
                _prefetch_schedule_cache_for_year(
                    team_names=season_df["Team"].dropna().astype(str).tolist(),
                    browser=browser,
                    year=year,
                    cache_dir=schedule_cache_dir,
                    force_refresh=schedule_force_refresh,
                    progress=schedule_progress,
                )
        else:
            warnings.warn(f"Skipping KenPom season {year} - data unavailable.", stacklevel=2)
    return result


# ---------------------------------------------------------------------------
# Conference-strength adjustment and sigma
# ---------------------------------------------------------------------------

def _apply_conf_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust AdjO/AdjD slightly for conference strength."""

    df = df.copy()
    df["_NetRtg"] = df["AdjO"] - df["AdjD"]
    global_avg = df["_NetRtg"].mean()
    conf_avg = df.groupby("Conf")["_NetRtg"].transform("mean")
    adj_factor = CONF_ADJ_WEIGHT * (conf_avg - global_avg)
    df["AdjO"] = df["AdjO"] + adj_factor * 0.5
    df["AdjD"] = df["AdjD"] - adj_factor * 0.5
    return df.drop(columns=["_NetRtg"])


def _compute_sigma(win_pct: float) -> float:
    """Estimate team-level game-to-game variance from win percentage."""

    raw = BASELINE_SIGMA * 4.0 * win_pct * (1.0 - win_pct)
    return float(np.clip(raw, 0.1, 0.5))


# ---------------------------------------------------------------------------
# Schedule loading and feature extraction
# ---------------------------------------------------------------------------

def _safe_team_filename(team_name: str) -> str:
    return team_name.replace("/", "-").replace("\\", "-").replace(":", "-")


def _normalize_schedule_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in _SCHEDULE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[_SCHEDULE_COLUMNS].copy()
    for col in _SCHEDULE_COLUMNS:
        df[col] = df[col].fillna("")
    return df


def fetch_kenpom_schedule(
    browser,
    team: str,
    year: int,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch one team's KenPom schedule via kenpompy, or load it from cache.

    Cached files are stored at `{cache_dir}/{year}/{team}.csv`.
    """

    cache_path = Path(cache_dir) / str(year) / f"{_safe_team_filename(team)}.csv"
    if cache_path.exists() and not force_refresh:
        return _normalize_schedule_frame(pd.read_csv(cache_path))

    if browser is None:
        warnings.warn(
            f"Missing browser for uncached KenPom schedule fetch: {team} ({year}).",
            stacklevel=2,
        )
        return None

    try:
        from kenpompy.team import get_schedule

        schedule_df = get_schedule(browser, team=team, season=str(year))
        schedule_df = _normalize_schedule_frame(schedule_df)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_df.to_csv(cache_path, index=False)
        return schedule_df
    except Exception as exc:
        warnings.warn(
            f"Could not fetch KenPom schedule for '{team}' ({year}): {exc}",
            stacklevel=2,
        )
        return None


def _prefetch_schedule_cache_for_year(
    team_names: list[str],
    browser,
    year: int,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> None:
    """
    Ensure all schedule CSVs for one season exist locally.

    Cached files are left untouched unless `force_refresh=True`.
    Missing schedules are fetched sequentially to avoid hammering KenPom.
    """

    missing_teams: list[str] = []
    cached_count = 0

    for team in team_names:
        cache_path = Path(cache_dir) / str(year) / f"{_safe_team_filename(team)}.csv"
        if cache_path.exists() and not force_refresh:
            cached_count += 1
        else:
            missing_teams.append(team)

    total = len(team_names)
    if progress:
        print(
            f"Year {year}: schedule cache ready for {cached_count}/{total}; "
            f"fetching {len(missing_teams)} missing."
        )

    if not missing_teams:
        return

    if browser is None:
        warnings.warn(
            f"Cannot prefetch missing KenPom schedules for {year} without a browser login "
            f"({len(missing_teams)} schedule files still missing).",
            stacklevel=2,
        )
        return

    for index, team in enumerate(missing_teams, start=1):
        fetch_kenpom_schedule(
            browser=browser,
            team=team,
            year=year,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        time.sleep(1.5)
        if progress and (index == len(missing_teams) or index % 25 == 0):
            print(f"  fetched {index}/{len(missing_teams)} schedules for {year}.")


def _parse_win_flag(result: str) -> Optional[int]:
    match = re.match(r"^\s*([WL])", str(result).strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return 1 if match.group(1).upper() == "W" else 0


def _parse_rank(rank_value: object, default_rank: int = DEFAULT_UNRANKED_OPPONENT) -> int:
    match = re.search(r"(\d+)", str(rank_value))
    if match is None:
        return default_rank
    return max(1, min(int(match.group(1)), default_rank))


def _opponent_weight(rank: int, max_rank: int = DEFAULT_UNRANKED_OPPONENT) -> float:
    rank = max(1, min(rank, max_rank))
    return 1.0 + (max_rank - rank) / max_rank


def _normalize_postseason_label(value: object) -> str:
    return str(value).strip().upper()


def _parse_schedule_sort_key(date_value: object) -> Optional[int]:
    """
    Build a season-aware sort key from KenPom's yearless date strings.

    KenPom schedules use values like "Fri Nov 13" and span Nov-Mar.
    We map Nov/Dec before Jan-Apr so regular-season order is preserved.
    """

    match = re.search(r"\b([A-Z][a-z]{2})\.?\s+(\d{1,2})\b", str(date_value).strip())
    if match is None:
        return None

    month = _MONTH_TO_NUM.get(match.group(1))
    if month is None:
        return None

    day = int(match.group(2))
    season_month = month if month >= 11 else month + 12
    return season_month * 100 + day


def _filter_pre_ncaa_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_schedule_frame(schedule_df)
    label = df["Postseason"].map(_normalize_postseason_label)
    exclude_mask = np.zeros(len(df), dtype=bool)
    for prefix in _POSTSEASON_EXCLUDE_PREFIXES:
        exclude_mask |= label.str.startswith(prefix).to_numpy()
    return df.loc[~exclude_mask].copy()


def _prepare_played_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    games = _filter_pre_ncaa_games(schedule_df)
    games["win_flag"] = games["Result"].map(_parse_win_flag)
    games = games.dropna(subset=["win_flag"]).copy()

    games["DateSortKey"] = games["Date"].map(_parse_schedule_sort_key)
    games["OpponentRankParsed"] = games["Opponent Rank"].map(_parse_rank)
    games["OpponentWeight"] = games["OpponentRankParsed"].map(_opponent_weight)
    games["win_flag"] = games["win_flag"].astype(int)

    if games["DateSortKey"].notna().any():
        games = games.sort_values(["DateSortKey", "Opponent Name"], na_position="last").reset_index(drop=True)
    else:
        games = games.reset_index(drop=True)
    return games


def compute_schedule_features(
    schedule_df: Optional[pd.DataFrame],
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> tuple[float, float]:
    """
    Derive two schedule-based features from pre-NCAA games.

    Returns:
        (recent_form, season_trajectory)

    recent_form:
        Opponent-weighted recent performance minus opponent-weighted season
        baseline. Positive means the team is outperforming its own season norm.

    season_trajectory:
        Slope of the centered rolling-form series over the season. Positive
        means the team's rolling performance trend improved into March.
    """

    if schedule_df is None or len(schedule_df) == 0:
        return 0.0, 0.0

    games = _prepare_played_games(schedule_df)
    if games.empty:
        return 0.0, 0.0

    weights = games["OpponentWeight"].to_numpy(dtype=float)
    results = games["win_flag"].to_numpy(dtype=float)
    window = max(1, min(int(recent_window), len(games)))

    season_baseline = float(np.average(results, weights=weights))

    recent_results = results[-window:]
    recent_weights = weights[-window:]
    recent_rate = float(np.average(recent_results, weights=recent_weights))
    recent_form = float(np.clip(recent_rate - season_baseline, -1.0, 1.0))

    rolling_deviation: list[float] = []
    for end_idx in range(len(games)):
        start_idx = max(0, end_idx - window + 1)
        rolling_rate = float(np.average(
            results[start_idx : end_idx + 1],
            weights=weights[start_idx : end_idx + 1],
        ))
        rolling_deviation.append(rolling_rate - season_baseline)

    if len(rolling_deviation) < 2:
        season_trajectory = 0.0
    else:
        slope = float(linregress(np.arange(len(rolling_deviation)), rolling_deviation).slope)
        season_trajectory = float(np.clip(slope * len(rolling_deviation), -1.0, 1.0))

    return recent_form, season_trajectory


def compute_conf_tourney_wins(schedule_df: Optional[pd.DataFrame]) -> int:
    """Count conference tournament wins prior to the NCAA tournament."""

    if schedule_df is None or len(schedule_df) == 0:
        return 0

    games = _normalize_schedule_frame(schedule_df)
    games["win_flag"] = games["Result"].map(_parse_win_flag)
    games = games.dropna(subset=["win_flag"]).copy()
    if games.empty:
        return 0

    labels = games["Postseason"].map(_normalize_postseason_label)
    valid_postseason = labels != ""
    for prefix in _POSTSEASON_EXCLUDE_PREFIXES:
        valid_postseason &= ~labels.str.startswith(prefix)

    return int(games.loc[valid_postseason, "win_flag"].sum())


def build_schedule_context_map(
    team_names: list[str],
    browser,
    year: int,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
    request_delay: float = 1.5,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """
    Fetch schedules for one season and build:
      - schedule feature map: {team: (recent_form, season_trajectory)}
      - conf tournament wins map: {team: wins}

    request_delay: seconds to sleep between actual network fetches (cache hits
    are free). Increase if you see 429 errors from KenPom.
    """

    feature_map: dict[str, tuple[float, float]] = {}
    conf_tourney_map: dict[str, int] = {}
    total = len(team_names)

    for index, team in enumerate(team_names, start=1):
        cache_path = Path(cache_dir) / str(year) / f"{_safe_team_filename(team)}.csv"
        is_cached = cache_path.exists() and not force_refresh

        schedule_df = fetch_kenpom_schedule(
            browser=browser,
            team=team,
            year=year,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        feature_map[team] = compute_schedule_features(schedule_df, recent_window=recent_window)
        conf_tourney_map[team] = compute_conf_tourney_wins(schedule_df)

        if not is_cached and request_delay > 0:
            time.sleep(request_delay)

        if progress and (index == total or index % 25 == 0):
            print(f"Processed {index}/{total} KenPom schedules for {year}.")

    return feature_map, conf_tourney_map


def build_schedule_features_map(
    team_names: list[str],
    browser,
    year: int,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> dict[str, tuple[float, float]]:
    """Convenience wrapper that returns only the schedule feature map."""

    feature_map, _ = build_schedule_context_map(
        team_names=team_names,
        browser=browser,
        year=year,
        recent_window=recent_window,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        progress=progress,
    )
    return feature_map


def build_conf_tourney_results_map(
    team_names: list[str],
    browser,
    year: int,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> dict[str, int]:
    """Convenience wrapper that returns only conference tournament wins."""

    _, conf_tourney_map = build_schedule_context_map(
        team_names=team_names,
        browser=browser,
        year=year,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        progress=progress,
    )
    return conf_tourney_map


def build_historical_schedule_context(
    kenpom_stats: dict[int, pd.DataFrame],
    browser=None,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, dict[str, int]]]:
    """Build cached schedule features and conference tournament wins for many seasons."""

    feature_history: dict[int, dict[str, tuple[float, float]]] = {}
    conf_history: dict[int, dict[str, int]] = {}

    for year in sorted(kenpom_stats):
        team_names = kenpom_stats[year]["Team"].tolist()
        feature_map, conf_map = build_schedule_context_map(
            team_names=team_names,
            browser=browser,
            year=year,
            recent_window=recent_window,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
            progress=progress,
        )
        feature_history[year] = feature_map
        conf_history[year] = conf_map

    return feature_history, conf_history


def build_historical_schedule_features(
    kenpom_stats: dict[int, pd.DataFrame],
    browser=None,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Convenience wrapper that returns only historical schedule features."""

    feature_history, _ = build_historical_schedule_context(
        kenpom_stats=kenpom_stats,
        browser=browser,
        recent_window=recent_window,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        progress=progress,
    )
    return feature_history


def build_historical_conf_tourney_results(
    kenpom_stats: dict[int, pd.DataFrame],
    browser=None,
    cache_dir: str = "data/kenpom/schedules",
    force_refresh: bool = False,
    progress: bool = True,
) -> dict[int, dict[str, int]]:
    """Convenience wrapper that returns only historical conference tournament wins."""

    _, conf_history = build_historical_schedule_context(
        kenpom_stats=kenpom_stats,
        browser=browser,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        progress=progress,
    )
    return conf_history


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_team_features(
    kenpom_df: pd.DataFrame,
    seeds: dict[str, tuple[int, str]],
    schedule_features_map: Optional[dict[str, tuple[float, float]]] = None,
    conf_tourney_results: Optional[dict[str, int]] = None,
    kaggle_team_features: Optional[dict[str, dict]] = None,
) -> dict[str, TeamFeatures]:
    """
    Build TeamFeatures objects for every seeded team found in the KenPom data.

    Args:
        kaggle_team_features: {kenpom_team_name: {"neutral_win_pct": float,
                              "coach_tourney_wins": int, ...}} — pre-resolved
                              Kaggle features keyed by KenPom name.
                              Build this from build_kaggle_features() output
                              using your TeamID->name mapping, then pass here.
                              When None, Kaggle features default to neutral values.
    """

    df = _apply_conf_adjustment(kenpom_df)
    kenpom_index = df.set_index("Team")

    not_found = [team for team in seeds if team not in kenpom_index.index]
    if not_found:
        warnings.warn(
            "The following teams are in seeds but not in the KenPom data:\n"
            f"  {not_found}\n"
            "Add them to name_map to fix mismatches.",
            stacklevel=2,
        )

    features: dict[str, TeamFeatures] = {}
    for team, (seed, region) in seeds.items():
        if team not in kenpom_index.index:
            continue

        row = kenpom_index.loc[team]
        recent_form, season_trajectory = (0.0, 0.0)
        if schedule_features_map is not None:
            recent_form, season_trajectory = schedule_features_map.get(team, (0.0, 0.0))

        conf_wins = 0
        if conf_tourney_results is not None:
            conf_wins = int(conf_tourney_results.get(team, 0))

        neutral_win_pct = 0.5
        tourney_coach_wins = 0
        if kaggle_team_features is not None and team in kaggle_team_features:
            kf = kaggle_team_features[team]
            nwp = kf.get("neutral_win_pct")
            cwins = kf.get("coach_tourney_wins")
            if nwp is not None and pd.notna(nwp):
                neutral_win_pct = float(nwp)
            if cwins is not None and pd.notna(cwins):
                tourney_coach_wins = int(cwins)

        def _ff(col: str) -> float:
            return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0

        features[team] = TeamFeatures(
            name=team,
            adj_o=float(row["AdjO"]),
            adj_d=float(row["AdjD"]),
            adj_t=float(row["AdjT"]),
            luck=float(row["Luck"]),
            win_pct=float(row["WinPct"]),
            conf=str(row["Conf"]),
            seed=seed,
            region=region,
            recent_form=float(recent_form),
            season_trajectory=float(season_trajectory),
            conf_tourney_wins=conf_wins,
            sigma=_compute_sigma(float(row["WinPct"])),
            efg_o=_ff("EFG_O"),
            efg_d=_ff("EFG_D"),
            tov_o=_ff("TOV_O"),
            tov_d=_ff("TOV_D"),
            orb=_ff("ORB"),
            drb=_ff("DRB"),
            ftr_o=_ff("FTR_O"),
            ftr_d=_ff("FTR_D"),
            neutral_win_pct=neutral_win_pct,
            tourney_coach_wins=tourney_coach_wins,
        )

    return features


# ---------------------------------------------------------------------------
# Kaggle historical data
# ---------------------------------------------------------------------------

def load_kaggle_historical(results_path: str, seeds_path: str) -> pd.DataFrame:
    """
    Load Kaggle March Machine Learning Mania tournament results joined with seeds.

    Filters to DayNum >= 136 so only main-bracket NCAA games remain.
    """

    results = pd.read_csv(results_path)
    seeds = pd.read_csv(seeds_path)

    seeds["SeedInt"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    seed_map = seeds.set_index(["Season", "TeamID"])["SeedInt"].to_dict()

    results["WSeed"] = results.apply(
        lambda row: seed_map.get((row["Season"], row["WTeamID"])),
        axis=1,
    )
    results["LSeed"] = results.apply(
        lambda row: seed_map.get((row["Season"], row["LTeamID"])),
        axis=1,
    )

    results = results.dropna(subset=["WSeed", "LSeed"])
    results = results[results["DayNum"] >= 136].copy()
    results["WSeed"] = results["WSeed"].astype(int)
    results["LSeed"] = results["LSeed"].astype(int)

    return results[["Season", "WTeamID", "LTeamID", "WSeed", "LSeed", "DayNum"]]


# ---------------------------------------------------------------------------
# Bracket fetching from KenPom tourney.php
# ---------------------------------------------------------------------------

def fetch_bracket(
    browser,
    year: int,
    name_map: Optional[dict[str, str]] = None,
) -> tuple[dict[str, tuple[int, str]], dict[str, tuple[int, str]]]:
    """
    Scrape the NCAA tournament bracket from kenpom.com/tourney.php.

    Returns:
        main_bracket_teams: {team_name: (seed, region)}  — 64 teams
        first_four_teams:   {team_name: (seed, region)}  — up to 8 First Four teams

    The First Four teams are those whose seed appears twice in the same region
    (two teams competing for one bracket slot).

    name_map: optional {kenpom_name: bracket_name} overrides, applied to all
    returned names so they're consistent with your seed dict conventions.

    Raises an exception if the page cannot be fetched or parsed.
    """
    from bs4 import BeautifulSoup
    from kenpompy.misc import get_html

    url = f"https://kenpom.com/tourney.php?y={year}"
    html = get_html(browser, url)
    soup = BeautifulSoup(html, "html.parser")

    # KenPom tourney page has one table per region.
    # Each row: seed | team | ...
    # Section headers (e.g. "South Region") appear as colspan rows.
    main_bracket: dict[str, tuple[int, str]] = {}
    first_four:   dict[str, tuple[int, str]] = {}

    # Track (seed, region) -> list of teams to detect First Four duplicates
    slot_teams: dict[tuple[int, str], list[str]] = {}

    current_region = "Unknown"

    tables = soup.find_all("table")
    if not tables:
        raise ValueError(f"No tables found on {url} — page may not have loaded correctly.")

    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # Detect region header row (single wide cell containing "Region")
            if len(cells) == 1:
                txt = cells[0].get_text(strip=True)
                if "Region" in txt or "South" in txt or "East" in txt or "West" in txt or "Midwest" in txt:
                    # Extract just the region word
                    for region_word in ("South", "East", "West", "Midwest"):
                        if region_word in txt:
                            current_region = region_word
                            break
                continue

            # Data row: first cell = seed (numeric), second cell = team name
            try:
                seed = int(cells[0].get_text(strip=True))
            except ValueError:
                continue

            team_cell = cells[1]
            # Team name may be inside an <a> tag
            a_tag = team_cell.find("a")
            team_name = (a_tag.get_text(strip=True) if a_tag
                         else team_cell.get_text(strip=True))
            if not team_name:
                continue

            if name_map:
                team_name = name_map.get(team_name, team_name)

            slot = (seed, current_region)
            slot_teams.setdefault(slot, []).append(team_name)

    if not slot_teams:
        raise ValueError(
            f"Parsed 0 teams from {url}. "
            "KenPom tourney page may not yet be available for this season, "
            "or the page structure has changed."
        )

    for (seed, region), teams in slot_teams.items():
        if len(teams) == 1:
            main_bracket[teams[0]] = (seed, region)
        else:
            # Two teams share a slot — they're First Four opponents
            for t in teams:
                first_four[t] = (seed, region)

    return main_bracket, first_four


# ---------------------------------------------------------------------------
# Feature normalization (apply fitted scaler to TeamFeatures at sim time)
# ---------------------------------------------------------------------------

def normalize_team_features(
    features: dict[str, "TeamFeatures"],
    scaler: dict[str, dict[str, float]],
) -> dict[str, "TeamFeatures"]:
    """
    Apply a fitted feature scaler to a dict of TeamFeatures.

    Call this in the simulation notebook after build_team_features() and
    after loading the scaler produced by train.ipynb:

        scaler   = load_scaler("data/feature_scaler.json")
        features = build_team_features(kenpom_df, seeds, ...)
        features = normalize_team_features(features, scaler)

    Only the features present in `scaler` are touched — everything else
    (AdjO, AdjD, AdjT, sigma, seed, etc.) is left unchanged.

    Args:
        features: {team_name: TeamFeatures} from build_team_features().
        scaler:   {feature_name: {"mean": float, "std": float}}
                  produced by fit_feature_scaler() and saved with save_scaler().

    Returns a new dict with standardized TeamFeatures (original is not mutated).
    """
    from dataclasses import replace as _replace

    normalized: dict[str, TeamFeatures] = {}
    for team, tf in features.items():
        updates: dict[str, float] = {}
        for feat, stats in scaler.items():
            if hasattr(tf, feat):
                raw = getattr(tf, feat)
                z = (float(raw) - stats["mean"]) / stats["std"]
                updates[feat] = float(np.clip(z, -2.5, 2.5))
        normalized[team] = _replace(tf, **updates)
    return normalized


# ---------------------------------------------------------------------------
# Women's (and generic Kaggle) season stats — replaces KenPom for W pipeline
# ---------------------------------------------------------------------------

def _location_multiplier(loc: str) -> float:
    if loc == "H":
        return KAGGLE_HOME_EFF_MULTIPLIER
    if loc == "A":
        return 1.0 / KAGGLE_HOME_EFF_MULTIPLIER
    return 1.0


def _opponent_location(loc: str) -> str:
    if loc == "H":
        return "A"
    if loc == "A":
        return "H"
    return "N"


def _build_kaggle_team_game_rows(det: pd.DataFrame) -> pd.DataFrame:
    det = det.copy()
    det["WPoss"] = det["WFGA"] - det["WOR"] + det["WTO"] + POSSESSION_FTA_WEIGHT * det["WFTA"]
    det["LPoss"] = det["LFGA"] - det["LOR"] + det["LTO"] + POSSESSION_FTA_WEIGHT * det["LFTA"]
    det["GamePoss"] = ((det["WPoss"] + det["LPoss"]) / 2.0).clip(lower=1.0)

    def _per_side(is_winner: bool) -> pd.DataFrame:
        side = "W" if is_winner else "L"
        opp = "L" if is_winner else "W"
        loc_series = det["WLoc"] if is_winner else det["WLoc"].map(_opponent_location)
        team_loc_mult = loc_series.map(_location_multiplier).astype(float)
        opp_loc_mult = loc_series.map(lambda loc: _location_multiplier(_opponent_location(loc))).astype(float)
        fga = det[f"{side}FGA"].clip(lower=1)
        opp_fga = det[f"{opp}FGA"].clip(lower=1)
        poss = det[f"{side}Poss"].clip(lower=1.0)
        opp_poss = det[f"{opp}Poss"].clip(lower=1.0)
        game_poss = det["GamePoss"].to_numpy(dtype=float)
        off_eff_raw = det[f"{side}Score"].to_numpy(dtype=float) * 100.0 / game_poss
        def_eff_raw = det[f"{opp}Score"].to_numpy(dtype=float) * 100.0 / game_poss

        return pd.DataFrame(
            {
                "DayNum": det["DayNum"].to_numpy(dtype=int),
                "TeamID": det[f"{side}TeamID"].to_numpy(dtype=int),
                "OppID": det[f"{opp}TeamID"].to_numpy(dtype=int),
                "Loc": loc_series.to_numpy(dtype=str),
                "won": 1 if is_winner else 0,
                "GamePoss": game_poss,
                "OffEffNeutral": off_eff_raw / team_loc_mult.to_numpy(dtype=float),
                "DefEffNeutral": def_eff_raw / opp_loc_mult.to_numpy(dtype=float),
                "EFG_O": (
                    det[f"{side}FGM"].to_numpy(dtype=float) + 0.5 * det[f"{side}FGM3"].to_numpy(dtype=float)
                )
                / fga.to_numpy(dtype=float)
                * 100.0,
                "EFG_D": (
                    det[f"{opp}FGM"].to_numpy(dtype=float) + 0.5 * det[f"{opp}FGM3"].to_numpy(dtype=float)
                )
                / opp_fga.to_numpy(dtype=float)
                * 100.0,
                "TOV_O": det[f"{side}TO"].to_numpy(dtype=float) / poss.to_numpy(dtype=float) * 100.0,
                "TOV_D": det[f"{opp}TO"].to_numpy(dtype=float) / opp_poss.to_numpy(dtype=float) * 100.0,
                "ORB": det[f"{side}OR"].to_numpy(dtype=float)
                / (det[f"{side}OR"] + det[f"{opp}DR"]).clip(lower=1).to_numpy(dtype=float)
                * 100.0,
                "DRB": det[f"{side}DR"].to_numpy(dtype=float)
                / (det[f"{side}DR"] + det[f"{opp}OR"]).clip(lower=1).to_numpy(dtype=float)
                * 100.0,
                "FTR_O": det[f"{side}FTA"].to_numpy(dtype=float) / fga.to_numpy(dtype=float) * 100.0,
                "FTR_D": det[f"{opp}FTA"].to_numpy(dtype=float) / opp_fga.to_numpy(dtype=float) * 100.0,
            }
        )

    return pd.concat([_per_side(True), _per_side(False)], ignore_index=True)


def _iterative_adjusted_ratings(team_games: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    league_avg_off = float(team_games["OffEffNeutral"].mean())
    league_avg_tempo = float(team_games["GamePoss"].mean())

    raw_off = team_games.groupby("TeamID")["OffEffNeutral"].mean()
    raw_def = team_games.groupby("TeamID")["DefEffNeutral"].mean()
    raw_tempo = team_games.groupby("TeamID")["GamePoss"].mean()

    teams = raw_off.index.union(raw_def.index).union(raw_tempo.index)
    off_rating = raw_off.reindex(teams).fillna(league_avg_off).astype(float)
    def_rating = raw_def.reindex(teams).fillna(league_avg_off).astype(float)
    tempo_rating = raw_tempo.reindex(teams).fillna(league_avg_tempo).astype(float)

    team_ids = team_games["TeamID"].to_numpy(dtype=int)
    opp_ids = team_games["OppID"].to_numpy(dtype=int)
    off_neutral = team_games["OffEffNeutral"].to_numpy(dtype=float)
    def_neutral = team_games["DefEffNeutral"].to_numpy(dtype=float)
    raw_tempo_vals = team_games["GamePoss"].to_numpy(dtype=float)

    for _ in range(KAGGLE_RATING_ITERATIONS):
        opp_def = def_rating.reindex(opp_ids).to_numpy(dtype=float)
        opp_off = off_rating.reindex(opp_ids).to_numpy(dtype=float)
        team_tempo = tempo_rating.reindex(team_ids).to_numpy(dtype=float)
        opp_tempo = tempo_rating.reindex(opp_ids).to_numpy(dtype=float)

        adj_off_game = off_neutral * league_avg_off / np.clip(opp_def, 1.0, None)
        adj_def_game = def_neutral * league_avg_off / np.clip(opp_off, 1.0, None)
        expected_tempo = np.sqrt(np.clip(team_tempo, 1.0, None) * np.clip(opp_tempo, 1.0, None))
        adj_tempo_game = raw_tempo_vals * league_avg_tempo / np.clip(expected_tempo, 1.0, None)

        next_off = (
            pd.DataFrame({"TeamID": team_ids, "value": adj_off_game})
            .groupby("TeamID")["value"]
            .mean()
            .reindex(teams)
            .fillna(off_rating)
        )
        next_def = (
            pd.DataFrame({"TeamID": team_ids, "value": adj_def_game})
            .groupby("TeamID")["value"]
            .mean()
            .reindex(teams)
            .fillna(def_rating)
        )
        next_tempo = (
            pd.DataFrame({"TeamID": team_ids, "value": adj_tempo_game})
            .groupby("TeamID")["value"]
            .mean()
            .reindex(teams)
            .fillna(tempo_rating)
        )

        next_off *= league_avg_off / max(float(next_off.mean()), 1e-6)
        next_def *= league_avg_off / max(float(next_def.mean()), 1e-6)
        next_tempo *= league_avg_tempo / max(float(next_tempo.mean()), 1e-6)

        off_rating = KAGGLE_RATING_DAMPING * next_off + (1.0 - KAGGLE_RATING_DAMPING) * off_rating
        def_rating = KAGGLE_RATING_DAMPING * next_def + (1.0 - KAGGLE_RATING_DAMPING) * def_rating
        tempo_rating = KAGGLE_RATING_DAMPING * next_tempo + (1.0 - KAGGLE_RATING_DAMPING) * tempo_rating

    return off_rating, def_rating, tempo_rating


def _win_loss_strings(team_games: pd.DataFrame) -> pd.DataFrame:
    wins = team_games.groupby("TeamID")["won"].sum()
    games = team_games.groupby("TeamID")["won"].size()
    losses = games - wins
    out = pd.DataFrame({"Wins": wins, "Losses": losses}).fillna(0).astype(int)
    out["W-L"] = out["Wins"].astype(str) + "-" + out["Losses"].astype(str)
    out["WinPct"] = out["Wins"] / (out["Wins"] + out["Losses"]).clip(lower=1)
    return out

def load_kaggle_season_stats(
    seasons: list[int],
    kaggle_dir: str,
    gender: str = "W",
) -> dict[int, pd.DataFrame]:
    """
    Compute KenPom-like per-team season stats from Kaggle detailed results.

    The ratings are built from regular-season box scores only and iteratively
    opponent-adjusted, so they are closer to KenPom-style inputs than raw
    season averages while remaining fully reproducible from Kaggle data.

    Args:
        seasons:    List of season years to include.
        kaggle_dir: Path to directory containing Kaggle CSV files.
        gender:     ``"W"`` for women's (default) or ``"M"`` for men's.

    Returns:
        ``{season: DataFrame}`` with the same key columns expected elsewhere:
        Team, AdjO, AdjD, AdjT, Luck, WinPct, Conf,
        EFG_O, EFG_D, TOV_O, TOV_D, ORB, DRB, FTR_O, FTR_D, AdjEM.
    """
    pfx = gender.upper()
    kaggle_dir = Path(kaggle_dir)

    detailed  = pd.read_csv(kaggle_dir / f"{pfx}RegularSeasonDetailedResults.csv")
    teams_df  = pd.read_csv(kaggle_dir / f"{pfx}Teams.csv")
    confs_df  = pd.read_csv(kaggle_dir / f"{pfx}TeamConferences.csv")

    id_to_name: dict[int, str] = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    result: dict[int, pd.DataFrame] = {}
    for season in seasons:
        det = detailed[detailed["Season"] == season].copy()
        if det.empty:
            continue

        team_games = _build_kaggle_team_game_rows(det)
        off_rating, def_rating, tempo_rating = _iterative_adjusted_ratings(team_games)
        records = _win_loss_strings(team_games)

        agg = team_games.groupby("TeamID")[
            ["EFG_O", "EFG_D", "TOV_O", "TOV_D", "ORB", "DRB", "FTR_O", "FTR_D"]
        ].mean()
        agg["AdjO"] = off_rating
        agg["AdjD"] = def_rating
        agg["AdjT"] = tempo_rating
        agg["AdjEM"] = agg["AdjO"] - agg["AdjD"]
        agg = agg.join(records, how="left")

        pyth_win_pct = (agg["AdjO"].clip(lower=0.01) ** LUCK_PYTH_EXPONENT) / (
            (agg["AdjO"].clip(lower=0.01) ** LUCK_PYTH_EXPONENT)
            + (agg["AdjD"].clip(lower=0.01) ** LUCK_PYTH_EXPONENT)
        )
        agg["Luck"] = agg["WinPct"] - pyth_win_pct
        agg["Team"] = agg.index.map(id_to_name)
        agg = agg.dropna(subset=["Team"]).reset_index()

        conf_season = confs_df[confs_df["Season"] == season][["TeamID", "ConfAbbrev"]]
        agg = agg.merge(conf_season, on="TeamID", how="left")
        agg.rename(columns={"ConfAbbrev": "Conf"}, inplace=True)
        agg["Conf"] = agg["Conf"].fillna("ind")

        keep = [
            "Team",
            "AdjO",
            "AdjD",
            "AdjT",
            "Luck",
            "WinPct",
            "Conf",
            "EFG_O",
            "EFG_D",
            "TOV_O",
            "TOV_D",
            "ORB",
            "DRB",
            "FTR_O",
            "FTR_D",
            "AdjEM",
            "Wins",
            "Losses",
            "W-L",
        ]
        result[season] = agg[keep].copy()

    return result


def build_kaggle_schedule_features(
    seasons: list[int],
    kaggle_dir: str,
    gender: str = "W",
    recent_window: int = 10,
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, dict[str, int]]]:
    """
    Build schedule-derived features from Kaggle compact results.

    Replacement for ``build_historical_schedule_context()`` when KenPom
    schedules are unavailable (women's pipeline).

    Returns the same two-dict structure:
        schedule_features:   {season: {team_name: (recent_form, season_trajectory)}}
        conf_tourney_results:{season: {team_name: conf_tourney_wins}}

    ``recent_form`` = opponent-quality-weighted win rate over the last
    ``recent_window`` regular-season games, minus the season average.
    Opponent quality proxy: opponent season win percentage.

    ``season_trajectory`` is not computed here — returns 0.0 (it consistently
    trains to near-zero even in the men's model).
    """
    pfx = gender.upper()
    kaggle_dir = Path(kaggle_dir)

    compact   = pd.read_csv(kaggle_dir / f"{pfx}RegularSeasonCompactResults.csv")
    conf_df   = pd.read_csv(kaggle_dir / f"{pfx}ConferenceTourneyGames.csv")
    teams_df  = pd.read_csv(kaggle_dir / f"{pfx}Teams.csv")
    id_to_name: dict[int, str] = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    schedule_features:    dict[int, dict[str, tuple[float, float]]] = {}
    conf_tourney_results: dict[int, dict[str, int]] = {}

    for season in seasons:
        seas_cmp = compact[compact["Season"] == season].copy()
        if seas_cmp.empty:
            continue

        # ----- season win percentage for each team (used as quality proxy) -----
        wins   = seas_cmp.groupby("WTeamID").size().rename("wins")
        losses = seas_cmp.groupby("LTeamID").size().rename("losses")
        records = pd.concat([wins, losses], axis=1).fillna(0)
        records["total"]  = records["wins"] + records["losses"]
        records["winpct"] = records["wins"] / records["total"].clip(lower=1)
        winpct_map: dict[int, float] = records["winpct"].to_dict()

        # ----- per-team sorted game log ----------------------------------------
        # Build a flat list (day, team, won, opp_id) for every game
        rows_w = seas_cmp[["DayNum","WTeamID","LTeamID"]].copy()
        rows_w.columns = ["DayNum","TeamID","OppID"]
        rows_w["won"] = 1

        rows_l = seas_cmp[["DayNum","LTeamID","WTeamID"]].copy()
        rows_l.columns = ["DayNum","TeamID","OppID"]
        rows_l["won"] = 0

        gamelog = pd.concat([rows_w, rows_l], ignore_index=True).sort_values("DayNum")

        sf: dict[str, tuple[float, float]] = {}
        for team_id, grp in gamelog.groupby("TeamID"):
            team_name = id_to_name.get(int(team_id))
            if team_name is None:
                continue

            games = grp.sort_values("DayNum")
            season_winpct = winpct_map.get(int(team_id), 0.5)

            # Last N games
            last_n = games.tail(recent_window)
            if len(last_n) < 2:
                sf[team_name] = (0.0, 0.0)
                continue

            # Weighted win rate: weight = opponent season win%
            weights = last_n["OppID"].map(lambda oid: winpct_map.get(int(oid), 0.5))
            w_sum = float(weights.sum())
            if w_sum < 1e-6:
                w_sum = len(last_n)
                weights = pd.Series([1.0] * len(last_n), index=last_n.index)

            weighted_wins = float((last_n["won"].values * weights.values).sum())
            recent_winpct = weighted_wins / w_sum
            recent_form   = recent_winpct - season_winpct

            sf[team_name] = (float(np.clip(recent_form, -1.0, 1.0)), 0.0)

        schedule_features[season] = sf

        # ----- conference tournament wins --------------------------------------
        conf_seas = conf_df[conf_df["Season"] == season]
        ct: dict[str, int] = {}
        for _, row in conf_seas.iterrows():
            winner_name = id_to_name.get(int(row["WTeamID"]))
            if winner_name:
                ct[winner_name] = ct.get(winner_name, 0) + 1
        conf_tourney_results[season] = ct

    return schedule_features, conf_tourney_results
