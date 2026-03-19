"""
model.py - Win probability engine.

Core formula:
    p = sigmoid(effective_diff / sigma)

where effective_diff combines:
    1. Pace-adjusted offense-vs-defense matchup
    2. Heavy-tailed t-distribution shocks
    3. Recent form from the last 10 games
    4. Full-season trajectory from schedule trend
    5. Luck factor
    6. Conference tournament signal
    7. Seed-gap prior
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.stats import t as t_dist

from .data import TeamFeatures

# ---------------------------------------------------------------------------
# ModelParams
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    All tunable parameters in one place.

    Defaults are calibrated to reproduce historical seed-matchup upset rates.
    Override any field in the notebook to experiment.

    margin_scale is load-bearing: it converts expected points/100-possessions
    to the logistic scale. If you change it, all weight defaults need recalibration.
    """
    # Shock distribution (t-dist; calibrated vs Kaggle historical upset rates)
    shock_df: float = 5.0           # degrees of freedom — lower = fatter tails = more upsets
    shock_scale: float = 0.12       # scale of shocks relative to margin_scale

    # Secondary signal weights
    recent_form_weight: float = 0.20
    trajectory_weight: float = 0.0      # consistently zero in CV — not trained
    luck_weight: float = 0.08
    conf_tourney_weight: float = 0.05   # per conference tournament win
    seed_prior_weight: float = 0.0      # redundant with KenPom efficiency gap — not trained
    # Kaggle-derived feature weights (zeroed out in CV — redundant with KenPom AdjEM)
    neutral_win_pct_weight: float = 0.0
    coach_exp_weight: float = 0.0
    # Four-factor matchup weights (trained; added on top of AdjO/AdjD base margin)
    # Each weight scales the standardized net matchup advantage for that factor:
    #   efg_net = (efg_o_a - efg_d_b) - (efg_o_b - efg_d_a)  → positive = A shoots better
    #   to_net  = (tov_d_b - tov_o_a) - (tov_d_a - tov_o_b)  → positive = A protects ball
    #   orb_net = (orb_a - drb_b)     - (orb_b - drb_a)       → positive = A wins boards
    #   ftr_net = (ftr_o_a - ftr_d_b) - (ftr_o_b - ftr_d_a)  → positive = A draws more fouls
    w_efg: float = 0.0
    w_to: float = 0.0
    w_orb: float = 0.0
    w_ftr: float = 0.0

    # Fatigue
    fatigue_per_close_game: float = 0.03    # fatigue increment per close game
    fatigue_decay_between_rounds: float = 0.5   # 0=full reset, 1=full compound
    fatigue_cap: float = 1.25               # max fatigue multiplier (25% offense penalty)

    # League baselines (set from your KenPom CSV if they differ)
    league_avg_adj_o: float = 110.0     # league average offensive efficiency
    league_avg_adj_t: float = 68.0      # league average tempo (possessions/40min)

    # Scale factor: converts pts/100-possessions to logistic units.
    # Calibrated so that a typical 5v12 matchup (~10 pt/100poss difference)
    # produces ~65% win probability, matching historical upset rates.
    # Do NOT change this without recalibrating all weight defaults.
    margin_scale: float = 70.0

    # Post-hoc temperature scaling (tau > 1 softens probabilities, < 1 sharpens).
    # Applied after the model computes sigma_eff: effective denominator = sigma_eff * temperature.
    # Fitted separately via log-loss minimization AFTER the optimizer converges;
    # not included in the optimizer's parameter set so it cannot absorb signal issues.
    # A value of 1.0 (default) means no calibration applied.
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Historical seed-matchup upset rates (from Kaggle 1985–2023, post-64-team era)
# Keys: (lower_seed, higher_seed) i.e. (better, worse)
# Values: historical win rate of the lower (better) seed
# Used to build seed_gap_prior lookup.
# ---------------------------------------------------------------------------

HISTORICAL_SEED_WIN_RATES: dict[tuple[int, int], float] = {
    (1, 16): 0.990,
    (2, 15): 0.941,
    (3, 14): 0.848,
    (4, 13): 0.791,
    (5, 12): 0.648,
    (6, 11): 0.630,
    (7, 10): 0.598,
    (8,  9): 0.513,
}


def _logit(p: float) -> float:
    """log(p / (1-p)), clamped to avoid inf."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def seed_gap_prior(
    seed_a: Optional[int],
    seed_b: Optional[int],
    weight: float,
) -> float:
    """
    Additive adjustment based on seed matchup, scaled by `weight`.

    Uses a lookup table of logit(historical_win_rate) by seed pair.
    The logit is the additive adjustment needed so that, for two otherwise
    equal teams, sigmoid(seed_prior) matches historical upset frequency.

    Returns 0.0 if either seed is None (e.g. First Four placeholder).

    The result is bounded to ±2.0 (before weight scaling) to prevent the
    seed prior from completely dominating the efficiency-based model.
    """
    if seed_a is None or seed_b is None:
        return 0.0

    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)

    if lo == hi:
        return 0.0

    # Look up historical win rate for the lower seed
    rate = HISTORICAL_SEED_WIN_RATES.get((lo, hi))
    if rate is None:
        # Linear interpolation for seed pairs not in the table
        # (e.g. cross-region Final Four matchups)
        seed_diff = hi - lo
        # Rough estimate: each seed step ≈ 0.05 win rate advantage
        rate = min(0.5 + seed_diff * 0.04, 0.99)

    logit_val = float(np.clip(_logit(rate), -2.0, 2.0))

    # Flip sign if team_a is the worse seed
    if seed_a > seed_b:
        logit_val = -logit_val

    return weight * logit_val


# ---------------------------------------------------------------------------
# Core formula components
# ---------------------------------------------------------------------------

def compute_pace_adjusted_margin(
    features_a: TeamFeatures,
    features_b: TeamFeatures,
    params: ModelParams,
    fatigue_a: float = 1.0,
    fatigue_b: float = 1.0,
) -> float:
    """
    Pace-adjusted expected scoring margin using the KenPom efficiency framework.

    A's offense is discounted by B's defense relative to league average:
        a_pts_per_100 = (AdjO_a / fatigue_a) * (AdjD_b / league_avg_AdjO)

    The ratio AdjD_b / league_avg_AdjO captures how much B suppresses scoring:
        - AdjD_b = 95 (elite defense):  multiplier = 95/110 = 0.864
        - AdjD_b = 115 (poor defense):  multiplier = 115/110 = 1.045

    The final margin is scaled from pts/100-possessions to logistic units
    via params.margin_scale (default 10).

    Returns a float in roughly [-2, 2] for typical tournament matchups.
    """
    # Fatigue degrades offensive efficiency only
    adj_o_a = features_a.adj_o / fatigue_a
    adj_o_b = features_b.adj_o / fatigue_b

    # Each team's expected points per 100 possessions given opponent's defense
    a_pts = adj_o_a * (features_b.adj_d / params.league_avg_adj_o)
    b_pts = adj_o_b * (features_a.adj_d / params.league_avg_adj_o)

    raw_margin = a_pts - b_pts  # in pts/100-possession units

    # Pace factor: faster game → more possessions → margin matters more
    game_pace = (features_a.adj_t + features_b.adj_t) / 2.0
    pace_factor = game_pace / params.league_avg_adj_t

    return (raw_margin * pace_factor) / params.margin_scale


# ---------------------------------------------------------------------------
# Shared deterministic core (used by both win_probability variants)
# ---------------------------------------------------------------------------

def _deterministic_components(
    features_a: TeamFeatures,
    features_b: TeamFeatures,
    params: ModelParams,
    fatigue_a: float = 1.0,
    fatigue_b: float = 1.0,
) -> tuple[float, float]:
    """
    Compute the deterministic part of the win probability formula.

    Returns (mu, sigma) where:
        mu    = base_margin + recent_form + trajectory + luck + conf_tourney + seed_prior
        sigma = team-level uncertainty (clipped to [0.1, 0.5])

    The stochastic shocks are NOT included here — callers add them separately.
    This factoring lets win_probability() and win_probability_expected() share
    identical deterministic logic without duplication.
    """
    base_margin = compute_pace_adjusted_margin(
        features_a, features_b, params, fatigue_a, fatigue_b
    )

    recent_form_contrib = params.recent_form_weight * (
        features_a.recent_form - features_b.recent_form
    )

    trajectory_contrib = params.trajectory_weight * (
        features_a.season_trajectory - features_b.season_trajectory
    )

    luck_contrib = params.luck_weight * (features_a.luck - features_b.luck)

    conf_wins_contrib = params.conf_tourney_weight * (
        features_a.conf_tourney_wins - features_b.conf_tourney_wins
    )

    neutral_contrib = params.neutral_win_pct_weight * (
        features_a.neutral_win_pct - features_b.neutral_win_pct
    )

    coach_contrib = params.coach_exp_weight * (
        features_a.tourney_coach_wins - features_b.tourney_coach_wins
    )

    # Four-factor matchup: each term is the net advantage for team A over team B
    # in that factor dimension (A's offense vs B's defense, minus the symmetric).
    # Values are standardized (post-scaler) so weights are on a common scale.
    # Guard: if four factors are all zero (unavailable), contribution is zero.
    ff_contrib = 0.0
    if features_a.efg_o != 0.0 or features_b.efg_o != 0.0:
        efg_net = (features_a.efg_o - features_b.efg_d) - (features_b.efg_o - features_a.efg_d)
        to_net  = (features_b.tov_d - features_a.tov_o) - (features_a.tov_d - features_b.tov_o)
        orb_net = (features_a.orb   - features_b.drb)   - (features_b.orb   - features_a.drb)
        ftr_net = (features_a.ftr_o - features_b.ftr_d) - (features_b.ftr_o - features_a.ftr_d)
        ff_contrib = (
            params.w_efg * efg_net
            + params.w_to  * to_net
            + params.w_orb * orb_net
            + params.w_ftr * ftr_net
        )

    seed_contrib = seed_gap_prior(
        features_a.seed, features_b.seed, params.seed_prior_weight
    )

    mu = (
        base_margin
        + recent_form_contrib
        + trajectory_contrib
        + luck_contrib
        + conf_wins_contrib
        + neutral_contrib
        + coach_contrib
        + ff_contrib
        + seed_contrib
    )

    sigma = float(np.clip(
        (features_a.sigma + features_b.sigma) / 2.0,
        0.1, 0.5
    ))

    return mu, sigma


# ---------------------------------------------------------------------------
# Stochastic win probability (used at simulation time)
# ---------------------------------------------------------------------------

def win_probability(
    features_a: TeamFeatures,
    features_b: TeamFeatures,
    params: ModelParams,
    fatigue_a: float = 1.0,
    fatigue_b: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> float:
    """
    Probability that team_a beats team_b (stochastic — draws t-dist shocks).

    Use this during simulation. For training/optimization use
    win_probability_expected() instead.

    Returns float in (0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu, sigma = _deterministic_components(
        features_a, features_b, params, fatigue_a, fatigue_b
    )

    # Use the same sigma_eff as win_probability_expected so that the stochastic
    # and deterministic formulas are on a consistent scale.  Without this, shocks
    # drawn in pts/100-poss units are divided by the tiny team-level sigma (≈0.25)
    # rather than sigma_eff (≈1–7 depending on calibration), making every game random.
    safe_df = max(float(params.shock_df), 3.0)
    shock_var = 2.0 * (params.shock_scale ** 2) * safe_df / (safe_df - 2.0)
    sigma_eff = float(np.sqrt(sigma ** 2 + shock_var)) * max(params.temperature, 0.1)

    shock_a = float(t_dist.rvs(df=params.shock_df, scale=params.shock_scale,
                                random_state=rng))
    shock_b = float(t_dist.rvs(df=params.shock_df, scale=params.shock_scale,
                                random_state=rng))

    effective_diff = mu + (shock_a - shock_b)
    p = float(1.0 / (1.0 + np.exp(-np.clip(effective_diff / sigma_eff, -500, 500))))

    if verbose:
        base_margin = compute_pace_adjusted_margin(
            features_a, features_b, params, fatigue_a, fatigue_b
        )
        print(f"\n--- win_probability: {features_a.name} vs {features_b.name} ---")
        print(f"  base_margin:      {base_margin:+.4f}  "
              f"(AdjO {features_a.adj_o:.1f}/{features_b.adj_o:.1f}, "
              f"AdjD {features_a.adj_d:.1f}/{features_b.adj_d:.1f})")
        print(f"  shock net:        {shock_a - shock_b:+.4f}  "
              f"(a={shock_a:+.3f}, b={shock_b:+.3f})")
        print(f"  mu (det.):        {mu:+.4f}")
        print(f"  sigma_eff:        {sigma_eff:.4f}  (sigma={sigma:.4f}  shock_std={shock_var**0.5:.4f})")
        print(f"  effective_diff:   {effective_diff:+.4f}")
        print(f"  P({features_a.name} wins): {p:.4f}")

    return p


# ---------------------------------------------------------------------------
# Deterministic win probability (used for training / optimization)
# ---------------------------------------------------------------------------

def win_probability_expected(
    features_a: TeamFeatures,
    features_b: TeamFeatures,
    params: ModelParams,
    fatigue_a: float = 1.0,
    fatigue_b: float = 1.0,
) -> float:
    """
    Expected win probability — analytically integrates out the t-dist shocks.

    Why this instead of win_probability() for training:
        win_probability() draws fresh random shocks each call, making the loss
        surface noisy and impossible to optimize with gradient methods. This
        function gives the same *expected* probability without any sampling.

    Derivation:
        Let X = shock_a - shock_b, where shock_a, shock_b ~ t(df) * scale.
        X has variance: Var(X) = 2 * scale² * df/(df-2)  for df > 2.

        Using the normal approximation to marginalise the logistic over X:
            E[sigmoid((mu + X) / sigma)] ≈ sigmoid(mu / sigma_eff)
        where:
            sigma_eff = sqrt(sigma² + shock_var)
            shock_var  = 2 * scale² * df / (df - 2)

        This is exact for normal shocks and a close approximation for t-dist
        shocks (the approximation improves as df increases).

    For df <= 2 the t-dist has infinite variance; we cap the contribution at
    an effective df=3 floor to keep sigma_eff finite.

    Returns float in (0, 1).
    """
    mu, sigma = _deterministic_components(
        features_a, features_b, params, fatigue_a, fatigue_b
    )

    # Variance of (shock_a - shock_b); floor df at 3 to keep finite
    safe_df = max(float(params.shock_df), 3.0)
    shock_var = 2.0 * (params.shock_scale ** 2) * safe_df / (safe_df - 2.0)

    sigma_eff = float(np.sqrt(sigma ** 2 + shock_var)) * max(params.temperature, 0.1)
    p = float(1.0 / (1.0 + np.exp(-np.clip(mu / sigma_eff, -500, 500))))
    return p


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_seed_prior(kaggle_df) -> dict[tuple[int, int], float]:
    """
    Compute historical win rates for each seed matchup from Kaggle data.

    Args:
        kaggle_df: Output of load_kaggle_historical().

    Returns:
        {(lower_seed, higher_seed): win_rate_of_lower_seed}
        e.g. {(5, 12): 0.648, (8, 9): 0.513, ...}
    """
    rates: dict[tuple[int, int], dict] = {}

    for _, row in kaggle_df.iterrows():
        w_seed = int(row["WSeed"])
        l_seed = int(row["LSeed"])

        lo, hi = min(w_seed, l_seed), max(w_seed, l_seed)
        key = (lo, hi)

        if key not in rates:
            rates[key] = {"wins_by_lower": 0, "total": 0}

        rates[key]["total"] += 1
        if w_seed == lo:
            rates[key]["wins_by_lower"] += 1

    result: dict[tuple[int, int], float] = {}
    for key, counts in rates.items():
        if counts["total"] >= 5:   # only include pairs with enough history
            result[key] = counts["wins_by_lower"] / counts["total"]

    return result


def calibrate_shock_params(
    training_df,
    params: Optional["ModelParams"] = None,
    seed_matchup_pairs: Optional[list[tuple[int, int]]] = None,
    target_win_rates: Optional[dict[tuple[int, int], float]] = None,
) -> tuple[float, float, float]:
    """
    Jointly find (margin_scale, shock_df, shock_scale) that reproduce historical
    seed-matchup win rates using actual KenPom base_margins from training_df.

    Strategy:
        For each seed matchup (e.g. 5v12), compute the average *raw* margin
        (pts/100-poss, before dividing by margin_scale) across all historical games.
        Then find continuous (margin_scale, shock_df, shock_scale) such that:

            sigmoid(avg_raw_margin / margin_scale / sigma_eff) ≈ historical_win_rate

        where sigma_eff = sqrt(sigma_team² + 2 * shock_scale² * shock_df / (shock_df-2)).

        All three are needed together: margin_scale controls the overall KenPom
        signal strength (affects strong mismatches like 1v16 most), shock_scale
        controls how much randomness flattens the distribution (affects all
        matchups similarly), and shock_df controls tail fatness.

    Returns:
        (margin_scale, shock_df, shock_scale) — all continuous, use to update ModelParams.
    """
    from scipy.optimize import minimize as _minimize

    if params is None:
        params = ModelParams()

    if seed_matchup_pairs is None:
        seed_matchup_pairs = list(HISTORICAL_SEED_WIN_RATES.keys())

    if target_win_rates is None:
        target_win_rates = dict(HISTORICAL_SEED_WIN_RATES)

    p_ref = params.league_avg_adj_o

    # Pre-compute average *raw* margin (no margin_scale division) and sigma per matchup
    avg_raw_margins: dict[tuple[int, int], float] = {}
    avg_sigmas:      dict[tuple[int, int], float] = {}

    for pair in seed_matchup_pairs:
        lo, hi = pair
        mask = (training_df["seed_a"] == lo) & (training_df["seed_b"] == hi)
        sub  = training_df[mask]
        if sub.empty:
            mask2 = (training_df["seed_a"] == hi) & (training_df["seed_b"] == lo)
            sub   = training_df[mask2]
            if sub.empty:
                continue
            sign = -1.0
        else:
            sign = 1.0

        adj_o_a = sub["adj_o_a"].to_numpy(float)
        adj_o_b = sub["adj_o_b"].to_numpy(float)
        adj_d_a = sub["adj_d_a"].to_numpy(float)
        adj_d_b = sub["adj_d_b"].to_numpy(float)
        adj_t_a = sub["adj_t_a"].to_numpy(float)
        adj_t_b = sub["adj_t_b"].to_numpy(float)
        sigma_a = sub["sigma_a"].to_numpy(float)
        sigma_b = sub["sigma_b"].to_numpy(float)

        a_pts = adj_o_a * (adj_d_b / p_ref)
        b_pts = adj_o_b * (adj_d_a / p_ref)
        pace  = (adj_t_a + adj_t_b) / 2.0
        # Raw margin in pts/100-poss units — NOT divided by margin_scale yet
        raw_margins = sign * (a_pts - b_pts) * (pace / params.league_avg_adj_t)

        avg_raw_margins[pair] = float(raw_margins.mean())
        avg_sigmas[pair]      = float(np.clip((sigma_a + sigma_b) / 2.0, 0.1, 0.5).mean())

    pairs_with_data = [p for p in seed_matchup_pairs if p in avg_raw_margins]
    if not pairs_with_data:
        raise ValueError("No seed matchup data found in training_df.")

    def _sse(x: np.ndarray) -> float:
        margin_scale_val, shock_df_val, shock_scale_val = x
        safe_df   = max(float(shock_df_val), 2.1)
        shock_var = 2.0 * float(shock_scale_val) ** 2 * safe_df / (safe_df - 2.0)
        sse = 0.0
        for pair in pairs_with_data:
            mu        = avg_raw_margins[pair] / float(margin_scale_val)
            sigma_eff = float(np.sqrt(avg_sigmas[pair] ** 2 + shock_var))
            p_pred    = float(1.0 / (1.0 + np.exp(-mu / sigma_eff)))
            sse += (p_pred - target_win_rates[pair]) ** 2
        return sse

    result = _minimize(
        _sse,
        x0=[params.margin_scale, 5.0, 0.3],
        method="L-BFGS-B",
        bounds=[(1.0, 500.0), (2.1, 30.0), (0.001, 5.0)],
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    best_margin_scale = float(result.x[0])
    best_df           = float(result.x[1])
    best_scale        = float(result.x[2])

    print(
        f"Calibration complete: margin_scale={best_margin_scale:.2f}, "
        f"shock_df={best_df:.2f}, shock_scale={best_scale:.4f}  (SSE={result.fun:.6f})"
    )
    print("  Seed matchup fit:")
    safe_df   = max(best_df, 2.1)
    shock_var = 2.0 * best_scale ** 2 * safe_df / (safe_df - 2.0)
    for pair in pairs_with_data:
        mu        = avg_raw_margins[pair] / best_margin_scale
        sigma_eff = float(np.sqrt(avg_sigmas[pair] ** 2 + shock_var))
        p_fit     = float(1.0 / (1.0 + np.exp(-mu / sigma_eff)))
        print(f"    {pair[0]:2d}v{pair[1]:<2d}  fit={p_fit:.3f}  target={target_win_rates[pair]:.3f}")

    return best_margin_scale, best_df, best_scale


# ---------------------------------------------------------------------------
# Param persistence
# ---------------------------------------------------------------------------

def save_params(params: ModelParams, path: str) -> None:
    """Serialize ModelParams to a JSON file."""
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2)
    print(f"Params saved to {path}")


def load_params(path: str) -> ModelParams:
    """Load ModelParams from a JSON file produced by save_params()."""
    with open(path) as f:
        d = json.load(f)
    return ModelParams(**d)


def save_scaler(scaler: dict, path: str) -> None:
    """
    Serialize a feature scaler to JSON.

    `scaler` is the dict returned by fit_feature_scaler():
        {feature_name: {"mean": float, "std": float}}
    """
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"Scaler saved to {path}")


def load_scaler(path: str) -> dict:
    """Load a feature scaler from a JSON file produced by save_scaler()."""
    with open(path) as f:
        return json.load(f)
