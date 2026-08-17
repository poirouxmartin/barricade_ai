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


def evaluate(game, player):
    my_d = game.dist_to_goal(player)
    opp_d = game.dist_to_goal(1 - player)

    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        confidence = max(1.0 - CONF_W * game.walls_left[1 - player], 0.05)
        dist_adv *= confidence

    wall_adv = (game.walls_left[player] - game.walls_left[1 - player]) * WALL_W
    return dist_adv + wall_adv