"""Static evaluation for Barricade.

Score is from the perspective of `player` (higher = better for them).

Components:
  * distance advantage: opponent's BFS distance to goal minus our own;
  * confidence: the opponent's remaining walls scale how much we trust our
    distance lead — with zero walls left they can never lengthen our path
    (the lead is fully trusted), with many left the lead is discounted.
    A deficit is never discounted: falling behind stays dangerous, which
    keeps the engine from walking into traps;
  * wall parity: having more remaining walls than the opponent is a small
    bonus.

Weights (DIST_W, CONF_W, CONF_FLOOR) are tunable constants, shared with the
numba kernel (barricade/engine/kernel.py).
"""

DIST_W = 4.0
CONF_W = 0.12  # opponent-wall discount: conf = max(1 - CONF_W * opp_walls, CONF_FLOOR)
CONF_FLOOR = 0.3


def score(my_d, opp_d, my_walls, opp_walls):
    """Score from distances and wall counts. Higher is better for the player."""
    gap = (opp_d - my_d) * DIST_W
    if gap > 0:
        conf = max(1.0 - CONF_W * opp_walls, CONF_FLOOR)
        gap = int(gap * conf)
    return int(gap) + (my_walls - opp_walls)


def evaluate(game, player):
    return score(game.dist_to_goal(player), game.dist_to_goal(1 - player),
                game.walls_left[player], game.walls_left[1 - player])