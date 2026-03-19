"""
simulator.py — Monte Carlo simulation engine.

Two simulation modes:
    1. simulate_full_bracket()         — run N complete bracket simulations,
                                         returns champion counts.
    2. simulate_tournament_round_by_round() — simulate each game N times,
                                         advance the modal winner each round.

Fatigue model:
    - apply_fatigue() increases a winner's fatigue after each close game.
    - decay_fatigue() partially resets fatigue between rounds.
    - Fatigue is capped at params.fatigue_cap and never resets fully by default.

RNG:
    All randomness uses np.random.default_rng(seed), passed down the call stack.
    Two runs with the same seed produce identical results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace as dataclass_replace
from typing import Optional, Callable

import numpy as np

from .bracket import Game, next_round_matchups, round_label, REGION_ORDER
from .data import TeamFeatures
from .model import ModelParams, win_probability

# ---------------------------------------------------------------------------
# Single-game primitives
# ---------------------------------------------------------------------------

def simulate_game(
    game: Game,
    features: dict[str, TeamFeatures],
    params: ModelParams,
    fatigue: dict[str, float],
    rng: np.random.Generator,
    round_temperature_multipliers: Optional[dict[int, float]] = None,
) -> tuple[str, float]:
    """
    Simulate one game. Pure — does NOT mutate fatigue.

    Returns:
        (winner_name, p_team_a_wins)
    """
    f_a = fatigue.get(game.team_a, 1.0)
    f_b = fatigue.get(game.team_b, 1.0)

    ft_a = features.get(game.team_a)
    ft_b = features.get(game.team_b)

    if ft_a is None or ft_b is None:
        missing = game.team_a if ft_a is None else game.team_b
        raise KeyError(
            f"No features found for '{missing}'. "
            "Check that the team is in your seeds dict and KenPom CSV."
        )

    game_params = params
    if round_temperature_multipliers:
        round_multiplier = float(round_temperature_multipliers.get(game.round_number, 1.0))
        if round_multiplier != 1.0:
            game_params = dataclass_replace(
                params,
                temperature=float(max(params.temperature, 0.1) * max(round_multiplier, 0.1)),
            )

    p = win_probability(ft_a, ft_b, game_params,
                        fatigue_a=f_a, fatigue_b=f_b, rng=rng)

    winner = game.team_a if rng.random() < p else game.team_b
    return winner, p


def apply_fatigue(
    team: str,
    p_team_a: float,
    is_team_a: bool,
    fatigue: dict[str, float],
    params: ModelParams,
) -> None:
    """
    Update fatigue in place for the winning team.

    closeness = 1 - 2 * |p - 0.5|  (1 = 50/50 game, 0 = total blowout)
    Closer games cause more fatigue. Blowout wins cost almost nothing.
    """
    p = p_team_a if is_team_a else (1.0 - p_team_a)
    closeness = 1.0 - 2.0 * abs(p - 0.5)
    current = fatigue.get(team, 1.0)
    new_val = current * (1.0 + params.fatigue_per_close_game * closeness)
    fatigue[team] = min(new_val, params.fatigue_cap)


def decay_fatigue(
    fatigue: dict[str, float],
    params: ModelParams,
) -> dict[str, float]:
    """
    Apply between-round fatigue decay. Returns a new dict (immutable).

    With fatigue_decay_between_rounds = 0.5:
        fatigue of 1.20 → 1.10  (half the excess is removed)
    With decay = 0.0:  full reset to 1.0
    With decay = 1.0:  no decay (compounds forever — not recommended)
    """
    decay = params.fatigue_decay_between_rounds
    return {
        team: 1.0 + (val - 1.0) * decay
        for team, val in fatigue.items()
    }


# ---------------------------------------------------------------------------
# Single full-bracket run
# ---------------------------------------------------------------------------

def simulate_single_bracket(
    first_four_games: list[Game],
    round_of_64: list[Game],
    features: dict[str, TeamFeatures],
    params: ModelParams,
    rng: np.random.Generator,
    round_temperature_multipliers: Optional[dict[int, float]] = None,
) -> dict[str, list[str]]:
    """
    Run one complete bracket simulation.

    Returns {round_label: [winner_names]} for every round including First Four.
    """
    results: dict[str, list[str]] = {}
    fatigue: dict[str, float] = defaultdict(lambda: 1.0)

    # --- First Four ---
    if first_four_games:
        ff_winners: list[str] = []
        ff_by_slot: dict[tuple[int, str], str] = {}
        for game in first_four_games:
            winner, p = simulate_game(
                game,
                features,
                params,
                fatigue,
                rng,
                round_temperature_multipliers=round_temperature_multipliers,
            )
            apply_fatigue(winner, p, winner == game.team_a, fatigue, params)
            ff_winners.append(winner)
            # Map winning team's (seed, region) slot for substitution
            winning_features = features.get(winner)
            if winning_features and winning_features.seed and winning_features.region:
                ff_by_slot[(winning_features.seed, winning_features.region)] = winner
        results["First Four"] = ff_winners
        fatigue = decay_fatigue(fatigue, params)

        # Substitute First Four winners into Round of 64 matchups
        current_games = _substitute_first_four(round_of_64, ff_by_slot, features)
    else:
        current_games = round_of_64[:]

    # --- Main bracket rounds ---
    while current_games:
        label = round_label(len(current_games))
        winners: list[str] = []

        for game in current_games:
            winner, p = simulate_game(
                game,
                features,
                params,
                fatigue,
                rng,
                round_temperature_multipliers=round_temperature_multipliers,
            )
            apply_fatigue(winner, p, winner == game.team_a, fatigue, params)
            winners.append(winner)

        results[label] = winners

        if len(winners) == 1:
            results["Champion"] = winners[0]
            break

        fatigue = decay_fatigue(fatigue, params)
        current_games = next_round_matchups(winners, current_games)

    return results


def _substitute_first_four(
    round_of_64: list[Game],
    ff_by_slot: dict[tuple[int, str], str],
    features: dict[str, TeamFeatures],
) -> list[Game]:
    """
    Replace First Four placeholder teams in Round of 64 games with actual winners.
    A placeholder is any team whose (seed, region) matches a First Four slot.
    """
    if not ff_by_slot:
        return round_of_64

    updated: list[Game] = []
    for game in round_of_64:
        team_a = game.team_a
        team_b = game.team_b

        ft_a = features.get(team_a)
        ft_b = features.get(team_b)

        if ft_a and ft_a.seed and ft_a.region:
            team_a = ff_by_slot.get((ft_a.seed, ft_a.region), team_a)
        if ft_b and ft_b.seed and ft_b.region:
            team_b = ff_by_slot.get((ft_b.seed, ft_b.region), team_b)

        updated.append(Game(
            team_a=team_a,
            team_b=team_b,
            region=game.region,
            round_number=game.round_number,
        ))
    return updated


# ---------------------------------------------------------------------------
# Mode 1: Full Monte Carlo
# ---------------------------------------------------------------------------

def simulate_full_bracket(
    first_four_games: list[Game],
    round_of_64: list[Game],
    features: dict[str, TeamFeatures],
    params: ModelParams,
    n_sim: int = 10_000,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    round_temperature_multipliers: Optional[dict[int, float]] = None,
) -> dict[str, int]:
    """
    Monte Carlo simulation: run n_sim complete bracket simulations.

    Returns:
        {team_name: championship_count}

    Memory note: returns champion counts only (not full bracket paths).
    This keeps memory at O(68) rather than O(n_sim × 63 × team_name_length).

    Args:
        first_four_games:   From build_full_bracket() — may be empty list.
        round_of_64:        From build_full_bracket().
        features:           From build_team_features().
        params:             ModelParams instance.
        n_sim:              Number of complete bracket simulations.
        seed:               RNG seed for reproducibility. None = random.
        progress_callback:  Optional fn(sim_number, n_sim) called every 100 sims.
    """
    rng = np.random.default_rng(seed)
    champion_counts: dict[str, int] = defaultdict(int)

    bar_length = 50

    for sim in range(n_sim):
        if sim % 100 == 0 or sim == n_sim - 1:
            if progress_callback:
                progress_callback(sim, n_sim)
            else:
                progress = sim / n_sim
                filled = int(round(bar_length * progress))
                bar = "#" * filled + "-" * (bar_length - filled)
                print(f"\rSimulation Progress: [{bar}] {progress * 100:.1f}%", end="")

        bracket = simulate_single_bracket(
            first_four_games,
            round_of_64,
            features,
            params,
            rng,
            round_temperature_multipliers=round_temperature_multipliers,
        )
        champion = bracket.get("Champion")
        if champion:
            champion_counts[champion] += 1

    filled = bar_length
    print(f"\rSimulation Progress: [{'#' * filled}] 100.0%")
    return dict(champion_counts)


# ---------------------------------------------------------------------------
# Mode 2: Round-by-round (N sims per game, advance modal winner)
# ---------------------------------------------------------------------------

def simulate_tournament_round_by_round(
    first_four_games: list[Game],
    round_of_64: list[Game],
    features: dict[str, TeamFeatures],
    params: ModelParams,
    n_sim: int = 1_000,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    round_temperature_multipliers: Optional[dict[int, float]] = None,
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Round-by-round simulation: each game is simulated n_sim times independently.
    The team that wins the majority of simulations advances.

    Returns:
        round_results:  {round_label: [winner_names]}
        round_details:  {round_label: [(team_a, team_b, winner, win_count, avg_p,
                                        closeness, fatigue_after)]}

    This mode is more deterministic than the Monte Carlo mode — it finds the
    "most likely" path rather than sampling a distribution of paths.
    """
    rng = np.random.default_rng(seed)
    fatigue: dict[str, float] = defaultdict(lambda: 1.0)

    round_results: dict[str, list] = {}
    round_details: dict[str, list] = {}

    total_games = (len(first_four_games) + 63)  # 4 + 63
    games_done = 0
    bar_length = 50

    def _update_progress():
        nonlocal games_done
        games_done += 1
        if progress_callback:
            progress_callback(games_done, total_games)
        else:
            progress = games_done / total_games
            filled = int(round(bar_length * progress))
            bar = "#" * filled + "-" * (bar_length - filled)
            print(f"\rSimulation Progress: [{bar}] {progress * 100:.1f}%", end="")

    def _simulate_game_n_times(
        game: Game,
        current_fatigue: dict[str, float],
    ) -> tuple[str, int, float]:
        """Simulate one game n_sim times. Returns (winner, win_count, avg_p_team_a)."""
        win_counts: dict[str, int] = {game.team_a: 0, game.team_b: 0}
        p_values: list[float] = []

        for _ in range(n_sim):
            winner, p = simulate_game(
                game,
                features,
                params,
                current_fatigue,
                rng,
                round_temperature_multipliers=round_temperature_multipliers,
            )
            win_counts[winner] += 1
            p_values.append(p)

        best_team = max(win_counts, key=lambda t: win_counts[t])
        best_count = win_counts[best_team]
        avg_p = float(np.mean(p_values))
        return best_team, best_count, avg_p

    # --- First Four ---
    if first_four_games:
        ff_winners: list[str] = []
        ff_details: list[tuple] = []
        ff_by_slot: dict[tuple[int, str], str] = {}

        for game in first_four_games:
            winner, win_count, avg_p = _simulate_game_n_times(game, fatigue)
            # avg_p is P(team_a wins); adjust if team_b won
            p_winner = avg_p if winner == game.team_a else 1.0 - avg_p
            closeness = 1.0 - 2.0 * abs(p_winner - 0.5)
            apply_fatigue(winner, avg_p, winner == game.team_a, fatigue, params)

            ff_winners.append(winner)
            ff_details.append((
                game.team_a, game.team_b, winner,
                win_count, p_winner, closeness, fatigue.get(winner, 1.0)
            ))
            _update_progress()

            ft = features.get(winner)
            if ft and ft.seed and ft.region:
                ff_by_slot[(ft.seed, ft.region)] = winner

        round_results["First Four"] = ff_winners
        round_details["First Four"] = ff_details
        fatigue = decay_fatigue(fatigue, params)
        current_games = _substitute_first_four(round_of_64, ff_by_slot, features)
    else:
        current_games = round_of_64[:]

    # --- Main bracket rounds ---
    while current_games:
        label = round_label(len(current_games))
        winners: list[str] = []
        details: list[tuple] = []

        for game in current_games:
            winner, win_count, avg_p = _simulate_game_n_times(game, fatigue)
            p_winner = avg_p if winner == game.team_a else 1.0 - avg_p
            closeness = 1.0 - 2.0 * abs(p_winner - 0.5)
            apply_fatigue(winner, avg_p, winner == game.team_a, fatigue, params)

            winners.append(winner)
            details.append((
                game.team_a, game.team_b, winner,
                win_count, p_winner, closeness, fatigue.get(winner, 1.0)
            ))
            _update_progress()

        round_results[label] = winners
        round_details[label] = details

        if len(winners) == 1:
            round_results["Champion"] = winners[0]
            round_details["Champion"] = winners[0]
            break

        fatigue = decay_fatigue(fatigue, params)
        current_games = next_round_matchups(winners, current_games)

    print(f"\rSimulation Progress: [{'#' * bar_length}] 100.0%")
    return round_results, round_details
