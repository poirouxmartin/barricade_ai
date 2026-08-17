import random

from barricade.engine.base import Engine


class RandomEngine(Engine):
    def choose_move(self, game):
        return random.choice(game.legal_actions())