"""
kaggle_features.py - Per-team feature engineering from Kaggle March Mania data.

Produces a DataFrame indexed by (Season, TeamID) with columns:
    Four factors (raw, from regular season box scores):
        efg_o, efg_d          -- effective FG% offense / defense
        to_pct_o, to_pct_d    -- turnover rate offense / defense
        or_pct                -- offensive rebound rate
        dr_pct                -- defensive rebound rate
        ftr_o, ftr_d          -- free throw rate offense / defense
        threep_rate           -- 3PA / FGA (shot selection / variance proxy)

    Neutral court record:
        neutral_win_pct       -- win % on neutral courts during regular season
        neutral_games         -- number of neutral court games played

    Coach experience:
        coach_tourney_wins    -- career NCAA tournament wins (all prior seasons)
        coach_tourney_apps    -- career tournament appearances (team-seasons)

    Consensus ranking (pre-tournament snapshot, day ~133):
        net_rank              -- NET ordinal rank
        pom_rank              -- KenPom ordinal rank
        consensus_rank        -- median rank across all available systems
        rank_gap              -- NET rank minus POM rank (positive = committee rates lower)

All features are computed from data strictly before the tournament starts.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(kaggle_dir: str, filename: str) -> pd.DataFrame:
    path = Path(kaggle_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {kaggle_dir}. "
            "Download the Kaggle March Mania dataset and place files there."
        )
    return pd.read_csv(path)


def _team_season_stats(rs: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Aggregate regular-season box score stats per team for one season.

    Each game row has winner and loser stats. We compute team totals
    by stacking winner-perspective and loser-perspective rows.
    """
    df = rs[rs["Season"] == season].copy()

    # Winner perspective: team = W, opponent = L
    w = pd.DataFrame({
        "TeamID":   df["WTeamID"],
        "OppID":    df["LTeamID"],
        "loc":      df["WLoc"],
        "fgm":      df["WFGM"],  "fga":  df["WFGA"],
        "fgm3":     df["WFGM3"], "fga3": df["WFGA3"],
        "ftm":      df["WFTM"],  "fta":  df["WFTA"],
        "or_":      df["WOR"],   "dr_":  df["WDR"],
        "to_":      df["WTO"],
        "opp_fgm":  df["LFGM"],  "opp_fga":  df["LFGA"],
        "opp_fgm3": df["LFGM3"], "opp_fta":  df["LFTA"],
        "opp_or":   df["LOR"],   "opp_dr":   df["LDR"],
        "opp_to":   df["LTO"],
    })

    # Loser perspective: team = L, opponent = W
    # Flip location: H→A, A→H, N→N
    loc_flip = {"H": "A", "A": "H", "N": "N"}
    l = pd.DataFrame({
        "TeamID":   df["LTeamID"],
        "OppID":    df["WTeamID"],
        "loc":      df["WLoc"].map(loc_flip),
        "fgm":      df["LFGM"],  "fga":  df["LFGA"],
        "fgm3":     df["LFGM3"], "fga3": df["LFGA3"],
        "ftm":      df["LFTM"],  "fta":  df["LFTA"],
        "or_":      df["LOR"],   "dr_":  df["LDR"],
        "to_":      df["LTO"],
        "opp_fgm":  df["WFGM"],  "opp_fga":  df["WFGA"],
        "opp_fgm3": df["WFGM3"], "opp_fta":  df["WFTA"],
        "opp_or":   df["WOR"],   "opp_dr":   df["WDR"],
        "opp_to":   df["WTO"],
    })

    games = pd.concat([w, l], ignore_index=True)
    return games


# ---------------------------------------------------------------------------
# Four factors
# ---------------------------------------------------------------------------

def compute_four_factors(
    rs: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Compute raw four-factors per team for one regular season.

    Returns DataFrame indexed by TeamID with columns:
        efg_o, efg_d, to_pct_o, to_pct_d,
        or_pct, dr_pct, ftr_o, ftr_d, threep_rate
    """
    games = _team_season_stats(rs, season)

    g = games.groupby("TeamID")

    fgm  = g["fgm"].sum()
    fga  = g["fga"].sum()
    fgm3 = g["fgm3"].sum()
    fga3 = g["fga3"].sum()
    fta  = g["fta"].sum()
    or_  = g["or_"].sum()
    dr_  = g["dr_"].sum()
    to_  = g["to_"].sum()

    opp_fgm  = g["opp_fgm"].sum()
    opp_fga  = g["opp_fga"].sum()
    opp_fgm3 = g["opp_fgm3"].sum()
    opp_fta  = g["opp_fta"].sum()
    opp_or   = g["opp_or"].sum()
    opp_dr   = g["opp_dr"].sum()
    opp_to   = g["opp_to"].sum()

    # Effective FG%: (FGM + 0.5*FGM3) / FGA
    efg_o = (fgm + 0.5 * fgm3) / fga.replace(0, np.nan)
    efg_d = (opp_fgm + 0.5 * opp_fgm3) / opp_fga.replace(0, np.nan)

    # Turnover rate: TO / (FGA + 0.475*FTA + TO)  [Dean Oliver]
    to_pct_o = to_ / (fga + 0.475 * fta + to_).replace(0, np.nan)
    to_pct_d = opp_to / (opp_fga + 0.475 * opp_fta + opp_to).replace(0, np.nan)

    # Offensive rebound %: OR / (OR + opponent DR)
    or_pct = or_ / (or_ + opp_dr).replace(0, np.nan)
    dr_pct = dr_ / (dr_ + opp_or).replace(0, np.nan)

    # Free throw rate: FTA / FGA
    ftr_o = fta / fga.replace(0, np.nan)
    ftr_d = opp_fta / opp_fga.replace(0, np.nan)

    # 3PA rate: 3PA / FGA  (shot selection / variance proxy)
    threep_rate = fga3 / fga.replace(0, np.nan)

    result = pd.DataFrame({
        "efg_o":      efg_o,
        "efg_d":      efg_d,
        "to_pct_o":   to_pct_o,
        "to_pct_d":   to_pct_d,
        "or_pct":     or_pct,
        "dr_pct":     dr_pct,
        "ftr_o":      ftr_o,
        "ftr_d":      ftr_d,
        "threep_rate": threep_rate,
    }).reset_index()
    result.insert(0, "Season", season)
    return result.set_index(["Season", "TeamID"])


# ---------------------------------------------------------------------------
# Neutral court record
# ---------------------------------------------------------------------------

def compute_neutral_court_record(
    rs: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Win % and game count on neutral courts during the regular season.

    Returns DataFrame indexed by TeamID with:
        neutral_win_pct, neutral_games
    """
    df = rs[(rs["Season"] == season) & (rs["WLoc"] == "N")].copy()

    wins  = df.groupby("WTeamID").size().rename("wins")
    losses = df.groupby("LTeamID").size().rename("losses")

    all_teams = pd.concat([wins, losses], axis=1).fillna(0)
    all_teams["neutral_games"]   = all_teams["wins"] + all_teams["losses"]
    all_teams["neutral_win_pct"] = all_teams["wins"] / all_teams["neutral_games"].replace(0, np.nan)

    result = all_teams[["neutral_win_pct", "neutral_games"]].reset_index()
    result.columns = ["TeamID", "neutral_win_pct", "neutral_games"]
    result.insert(0, "Season", season)
    return result.set_index(["Season", "TeamID"])


# ---------------------------------------------------------------------------
# Coach experience
# ---------------------------------------------------------------------------

def compute_coach_experience(
    coaches: pd.DataFrame,
    tourney_results: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Career NCAA tournament wins and appearances for each team's coach
    in `season`, counting only seasons BEFORE `season`.

    Returns DataFrame indexed by TeamID with:
        coach_tourney_wins, coach_tourney_apps
    """
    # Get coaches active in this season
    season_coaches = coaches[coaches["Season"] == season][["TeamID", "CoachName"]].copy()

    # All tournament games before this season
    past_tourney = tourney_results[tourney_results["Season"] < season]

    # Teams in each past tournament (appeared = had at least one game)
    # Win = won that game
    past_coaches = coaches[coaches["Season"] < season][["Season", "TeamID", "CoachName"]].copy()

    # Merge coach onto past wins
    win_rows = past_tourney[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    wins_with_coach = win_rows.merge(past_coaches, on=["Season", "TeamID"], how="left")
    coach_wins = (
        wins_with_coach.groupby("CoachName").size()
        .rename("coach_tourney_wins")
    )

    # Appearances = distinct (Season, TeamID) pairs per coach
    # A team appears once per tournament regardless of how far they go
    app_rows = pd.concat([
        past_tourney[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
        past_tourney[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
    ]).drop_duplicates()
    apps_with_coach = app_rows.merge(past_coaches, on=["Season", "TeamID"], how="left")
    # Count distinct (Season, TeamID) per coach = team-season appearances
    coach_apps = (
        apps_with_coach.drop_duplicates(["Season", "TeamID", "CoachName"])
        .groupby("CoachName").size()
        .rename("coach_tourney_apps")
    )

    # Attach to current season teams
    result = season_coaches.copy()
    result = result.merge(coach_wins.reset_index(), on="CoachName", how="left")
    result = result.merge(coach_apps.reset_index(), on="CoachName", how="left")
    result[["coach_tourney_wins", "coach_tourney_apps"]] = (
        result[["coach_tourney_wins", "coach_tourney_apps"]].fillna(0)
    )
    result.insert(0, "Season", season)
    return result[["Season", "TeamID", "coach_tourney_wins", "coach_tourney_apps"]].set_index(["Season", "TeamID"])


# ---------------------------------------------------------------------------
# Consensus ranking
# ---------------------------------------------------------------------------

_RANKING_SYSTEMS = {
    "NET": "net_rank",
    "POM": "pom_rank",
    "MAS": "mas_rank",
    "SAG": "sag_rank",
    "AP":  "ap_rank",
}

def compute_consensus_ranking(
    massey: pd.DataFrame,
    season: int,
    pre_tourney_day: int = 133,
    systems: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Pre-tournament ranking consensus from MMasseyOrdinals.

    Uses the snapshot closest to `pre_tourney_day` (selection Sunday ~day 133).
    For each team, computes:
        net_rank, pom_rank, mas_rank, sag_rank, ap_rank  (individual systems)
        consensus_rank  -- median rank across all available systems
        rank_gap        -- NET rank minus POM rank
                          positive = committee ranks team lower than efficiency model

    Returns DataFrame indexed by TeamID.
    """
    if systems is None:
        systems = _RANKING_SYSTEMS

    df = massey[massey["Season"] == season].copy()

    # Use the latest available snapshot at or before pre_tourney_day
    available_days = sorted(df["RankingDayNum"].unique())
    snap_day = max((d for d in available_days if d <= pre_tourney_day), default=None)
    if snap_day is None:
        snap_day = available_days[0]
        warnings.warn(
            f"No Massey ranking snapshot at or before day {pre_tourney_day} for {season}. "
            f"Using day {snap_day} instead.",
            stacklevel=2,
        )

    snap = df[df["RankingDayNum"] == snap_day]

    # Pivot: one row per team, one column per system
    pivot = (
        snap[snap["SystemName"].isin(systems)]
        .pivot_table(index="TeamID", columns="SystemName", values="OrdinalRank", aggfunc="first")
        .rename(columns=systems)
    )

    # Consensus rank = median across ALL available systems at this snapshot
    all_systems = snap.pivot_table(
        index="TeamID", columns="SystemName", values="OrdinalRank", aggfunc="first"
    )
    pivot["consensus_rank"] = all_systems.median(axis=1)

    # Rank gap: NET - POM  (+ = committee thinks team is worse than efficiency says)
    if "net_rank" in pivot.columns and "pom_rank" in pivot.columns:
        pivot["rank_gap"] = pivot["net_rank"] - pivot["pom_rank"]
    else:
        pivot["rank_gap"] = np.nan

    pivot.insert(0, "Season", season)
    return pivot.reset_index().set_index(["Season", "TeamID"])


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_kaggle_features(
    season: int,
    kaggle_dir: str = "data/kaggle",
    pre_tourney_day: int = 133,
) -> pd.DataFrame:
    """
    Build all Kaggle-derived features for one season.

    Returns a DataFrame indexed by (Season, TeamID) containing:
        Four factors, neutral court record, coach experience,
        consensus rankings, rank gap.

    All features are computed from data strictly before the tournament.
    """
    rs       = _load(kaggle_dir, "MRegularSeasonDetailedResults.csv")
    coaches  = _load(kaggle_dir, "MTeamCoaches.csv")
    tourney  = _load(kaggle_dir, "MNCAATourneyCompactResults.csv")
    massey   = _load(kaggle_dir, "MMasseyOrdinals.csv")

    four_factors = compute_four_factors(rs, season)
    neutral      = compute_neutral_court_record(rs, season)
    coach_exp    = compute_coach_experience(coaches, tourney, season)
    rankings     = compute_consensus_ranking(massey, season, pre_tourney_day)

    result = four_factors \
        .join(neutral,   how="left") \
        .join(coach_exp, how="left") \
        .join(rankings,  how="left")

    return result


def build_kaggle_features_historical(
    seasons: list[int],
    kaggle_dir: str = "data/kaggle",
    pre_tourney_day: int = 133,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build Kaggle features for multiple seasons and concatenate.

    Loads each large file once and passes to per-season functions.
    """
    rs      = _load(kaggle_dir, "MRegularSeasonDetailedResults.csv")
    coaches = _load(kaggle_dir, "MTeamCoaches.csv")
    tourney = _load(kaggle_dir, "MNCAATourneyCompactResults.csv")
    massey  = _load(kaggle_dir, "MMasseyOrdinals.csv")

    frames: list[pd.DataFrame] = []
    for season in seasons:
        four_factors = compute_four_factors(rs, season)
        neutral      = compute_neutral_court_record(rs, season)
        coach_exp    = compute_coach_experience(coaches, tourney, season)
        rankings     = compute_consensus_ranking(massey, season, pre_tourney_day)

        frame = four_factors \
            .join(neutral,   how="left") \
            .join(coach_exp, how="left") \
            .join(rankings,  how="left")
        frames.append(frame)

        if verbose:
            print(f"  Built Kaggle features for {season}: {len(frame)} teams")

    return pd.concat(frames)
