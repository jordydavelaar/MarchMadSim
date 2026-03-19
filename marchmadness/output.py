"""
output.py — Display and print functions.

All terminal formatting lives here so simulator.py stays clean.
"""

from __future__ import annotations

BOLD  = "\033[1m"
RESET = "\033[0m"
DIM   = "\033[2m"

ROUND_ORDER = [
    "First Four",
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "Championship",
]


def print_first_round_matchups(games: list) -> None:
    """Print Round of 64 matchups grouped by region."""
    from collections import defaultdict
    by_region: dict[str, list] = defaultdict(list)
    for game in games:
        region = game.region or "Unknown"
        by_region[region].append(game)

    print("Tournament Start")
    print("=" * 60)
    region_order = ["South", "East", "Midwest", "West"]
    for region in region_order:
        if region not in by_region:
            continue
        print(f"\n{BOLD}{region} Region{RESET}")
        print("-" * 40)
        for game in by_region[region]:
            ft_a_seed = ""
            ft_b_seed = ""
            print(f"  {game.team_a} vs {game.team_b}")


def print_round_results(
    round_name: str,
    details: list[tuple],
    n_sim: int,
) -> None:
    """
    Print one round's results.

    details format: (team_a, team_b, winner, win_count, avg_p, closeness, fatigue)
    """
    print(f"\n{BOLD}{round_name}{RESET}")
    print("-" * 60)
    for team_a, team_b, winner, win_count, avg_p, closeness, fatigue_val in details:
        pct = win_count / n_sim * 100
        loser = team_b if winner == team_a else team_a
        fatigue_str = f"{DIM}(fatigue={fatigue_val:.2f}){RESET}" if fatigue_val > 1.01 else ""
        print(
            f"  {team_a} vs {team_b}  →  "
            f"{BOLD}{winner}{RESET} "
            f"({win_count}/{n_sim}, {pct:.1f}%)  "
            f"{fatigue_str}"
        )


def print_champion_counts(
    champion_counts: dict[str, int],
    n_sim: int,
    top_n: int = 8,
) -> None:
    """Print top-N teams by championship probability."""
    print(f"\n{BOLD}Champion Probabilities (Top {top_n}){RESET}")
    print("=" * 50)
    sorted_champs = sorted(
        champion_counts.items(), key=lambda x: x[1], reverse=True
    )[:top_n]
    for i, (team, count) in enumerate(sorted_champs, 1):
        pct = count / n_sim * 100
        bar = "█" * int(pct / 2)
        print(f"  {i:2}. {BOLD}{team:<25}{RESET} {count:>5}/{n_sim}  {pct:5.1f}%  {bar}")


def print_full_tournament(
    round_results: dict,
    round_details: dict,
    n_sim: int,
) -> None:
    """Print all rounds sequentially."""
    print("\nTournament Results (round-by-round simulation)")
    print("=" * 60)

    for rnd in ROUND_ORDER:
        if rnd in round_details and isinstance(round_details[rnd], list):
            print_round_results(rnd, round_details[rnd], n_sim)

    champion = round_results.get("Champion") or round_details.get("Champion")
    if champion:
        print(f"\n{'=' * 60}")
        print(f"  {BOLD}CHAMPION: {champion}{RESET}")
        print(f"{'=' * 60}")
