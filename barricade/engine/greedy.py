"""Greedy engine: 1-ply, score = my dist to goal - opp dist to goal.

Minimizes its own distance while maximizing the opponent's. Placeholder
before the alpha-beta / NN engines.
"""

from barricade.engine.base import Engine


class GreedyEngine(Engine):
    def choose_move(self, game):
        player = game.turn
        best = None
        best_score = None
        for action in game.legal_actions():
            g = game.clone()
            g.apply(action)
            score = g.dist_to_goal(player) - g.dist_to_goal(1 - player)
            if best_score is None or score < best_score:
                best_score = score
                best = action
        return best