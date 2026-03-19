"""
bracket.py — Bracket structure and matchup generation.

Handles the full 68-team bracket including First Four games.
The public entry point is build_full_bracket().

Standard bracket seeding per region (16 teams):
    1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15

First Four structure (4 games before the Round of 64):
    - 2 games between 16-seeds competing for the 1v16 slot in two regions
    - 2 games between 11-seeds competing for the 6v11 slot in two regions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Game:
    """A single tournament game."""
    team_a: str
    team_b: str
    region: Optional[str]
    round_number: int           # 0 = First Four, 1 = Round of 64, ...
    is_first_four: bool = False

    def __repr__(self) -> str:
        tag = " [First Four]" if self.is_first_four else ""
        region = f" ({self.region})" if self.region else ""
        return f"<Game R{self.round_number}{tag}{region}: {self.team_a} vs {self.team_b}>"


# Standard seed matchup order within a region (index pairs into sorted seed list)
# Seeds sorted ascending: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# Indices:                  0 1 2 3 4 5 6 7 8  9  10 11 12 13 14 15
_SEED_MATCHUPS = [
    (0, 15),   # 1 vs 16
    (7, 8),    # 8 vs 9
    (4, 11),   # 5 vs 12
    (3, 12),   # 4 vs 13
    (5, 10),   # 6 vs 11
    (2, 13),   # 3 vs 14
    (6, 9),    # 7 vs 10
    (1, 14),   # 2 vs 15
]

REGION_ORDER = ["South", "East", "Midwest", "West"]

# ---------------------------------------------------------------------------
# First Four
# ---------------------------------------------------------------------------

def build_first_four_games(
    first_four_teams: dict[str, tuple[int, str]],
) -> list[Game]:
    """
    Build First Four games from a dict of play-in teams.

    Args:
        first_four_teams: {team_name: (seed, region)} for pending First Four teams.
                          Must be an even number of teams; each (seed, region) pair
                          must have exactly 2 teams.

    Returns:
        List of Game objects with round_number=0 and is_first_four=True.

    Raises:
        ValueError if the dict doesn't form valid pairs.
    """
    if len(first_four_teams) == 0 or len(first_four_teams) % 2 != 0:
        raise ValueError(
            f"first_four_teams must have an even number of teams, "
            f"got {len(first_four_teams)}."
        )

    # Group teams by (seed, region) — each group should have exactly 2 teams
    from collections import defaultdict
    slots: dict[tuple[int, str], list[str]] = defaultdict(list)
    for team, (seed, region) in first_four_teams.items():
        slots[(seed, region)].append(team)

    games: list[Game] = []
    for (seed, region), teams in sorted(slots.items()):
        if len(teams) != 2:
            raise ValueError(
                f"Expected exactly 2 teams for First Four slot "
                f"(seed={seed}, region={region}), got {teams}."
            )
        games.append(Game(
            team_a=teams[0],
            team_b=teams[1],
            region=region,
            round_number=0,
            is_first_four=True,
        ))

    return games


# ---------------------------------------------------------------------------
# Round of 64
# ---------------------------------------------------------------------------

def build_first_round_matchups(
    main_bracket_teams: dict[str, tuple[int, str]],
    first_four_winners: Optional[dict[tuple[int, str], str]] = None,
) -> list[Game]:
    """
    Generate the 32 Round of 64 matchups.

    Args:
        main_bracket_teams: {team_name: (seed, region)} for all 64 main-bracket teams.
        first_four_winners: {(seed, region): winner_name} — if provided, the winner
                            replaces the placeholder in that bracket slot.
                            (The losing First Four team has already been eliminated.)

    Returns:
        List of 32 Game objects with round_number=1.
    """
    # Group by region
    from collections import defaultdict
    by_region: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for team, (seed, region) in main_bracket_teams.items():
        by_region[region].append((seed, team))

    games: list[Game] = []
    for region in REGION_ORDER:
        teams_in_region = sorted(by_region[region], key=lambda x: x[0])

        if len(teams_in_region) != 16:
            raise ValueError(
                f"Region '{region}' has {len(teams_in_region)} teams, expected 16."
            )

        # teams_in_region[i] = (seed, team_name), sorted by seed ascending
        for idx_a, idx_b in _SEED_MATCHUPS:
            seed_a, team_a = teams_in_region[idx_a]
            seed_b, team_b = teams_in_region[idx_b]

            # Substitute First Four winner if applicable
            if first_four_winners:
                team_a = first_four_winners.get((seed_a, region), team_a)
                team_b = first_four_winners.get((seed_b, region), team_b)

            games.append(Game(
                team_a=team_a,
                team_b=team_b,
                region=region,
                round_number=1,
            ))

    return games


# ---------------------------------------------------------------------------
# Subsequent rounds
# ---------------------------------------------------------------------------

def next_round_matchups(winners: list[str], prev_games: list[Game]) -> list[Game]:
    """
    Pair winners into the next round's matchups.

    Winners are paired in order: (0,1), (2,3), (4,5), ...
    Region is inherited from the earlier game if both teams came from the same region,
    else set to None (cross-region games in Final Four / Championship).

    Args:
        winners:    Ordered list of winners from the previous round.
        prev_games: The games that produced those winners (same order).

    Returns:
        List of len(winners)//2 Game objects.
    """
    if len(winners) % 2 != 0:
        raise ValueError(
            f"Cannot pair an odd number of winners ({len(winners)})."
        )

    next_round = prev_games[0].round_number + 1 if prev_games else 1
    games: list[Game] = []

    for i in range(0, len(winners), 2):
        team_a = winners[i]
        team_b = winners[i + 1]

        # Determine region: same region for intra-region games, None for cross-region
        region_a = prev_games[i].region if i < len(prev_games) else None
        region_b = prev_games[i + 1].region if i + 1 < len(prev_games) else None
        region = region_a if region_a == region_b else None

        games.append(Game(
            team_a=team_a,
            team_b=team_b,
            region=region,
            round_number=next_round,
        ))

    return games


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_full_bracket(
    main_bracket_teams: dict[str, tuple[int, str]],
    first_four_teams: Optional[dict[str, tuple[int, str]]] = None,
) -> tuple[list[Game], list[Game]]:
    """
    Build the complete bracket structure.

    Args:
        main_bracket_teams: {team_name: (seed, region)} — the 64 main bracket teams.
                            If First Four winners are known in advance (rare), include
                            them here and omit first_four_teams.
        first_four_teams:   {team_name: (seed, region)} — the 8 First Four teams
                            (4 games). Optional; pass None to skip First Four.

    Returns:
        (first_four_games, round_of_64_games)
            first_four_games:  List of 4 First Four Game objects (empty if None passed).
            round_of_64_games: List of 32 Round of 64 Game objects.

    Note: The simulator handles advancing First Four winners into the Round of 64
    slots at runtime. This function just builds the static game structure.
    """
    first_four_games: list[Game] = []
    if first_four_teams:
        first_four_games = build_first_four_games(first_four_teams)

    round_of_64 = build_first_round_matchups(main_bracket_teams)

    return first_four_games, round_of_64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUND_LABELS = {
    32: "Round of 64",
    16: "Round of 32",
    8:  "Sweet 16",
    4:  "Elite 8",
    2:  "Final Four",
    1:  "Championship",
}

def round_label(n_games: int) -> str:
    """Return a human-readable label for a round given its number of games."""
    return ROUND_LABELS.get(n_games, f"Round ({n_games} games)")
