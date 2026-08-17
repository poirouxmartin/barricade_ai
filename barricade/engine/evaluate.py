"""Static evaluation for Barricade.

Score is from the perspective of `player` (higher = better for them).

Components:
  * distance advantage: opponent's BFS distance to goal minus our own;
  * confidence ratio: when our distance lead is positive, it is dampened by
    the opponent's remaining walls (their walls can still block us);
  * wall advantage: having more remaining walls than the opponent.

Weights (DIST_W, WALL_W, CONF_W) are tunable constants.
"""

DIST_W = 4.0
WALL_W = 1.0
CONF_W = 0.12  # confidence penalty per opponent wall remaining


def score(my_d, opp_d, my_walls, opp_walls):
    """Score from distances and wall counts. Higher is better for the player."""
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        confidence = max(1.0 - CONF_W * opp_walls, 0.05)
        dist_adv *= confidence
    return dist_adv + (my_walls - opp_walls) * WALL_W


def evaluate(game, player):
    return score(game.dist_to_goal(player), game.dist_to_goal(1 - player),
                game.walls_left[player], game.walls_left[1 - player])