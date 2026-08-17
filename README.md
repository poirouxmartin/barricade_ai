# Barricade

Jeu de stratégie 2 joueurs sur grille 9×9 (type Quoridor). Course vers le côté opposé du plateau, 10 barricades chacun, saut par-dessus l'adversaire.

## Lancer

```bash
python run.py          # http://127.0.0.1:8000
python -m unittest discover tests -v   # tests
```

Modes via "New game" : PvP, vs AI, AI vs AI.

## Structure

- `barricade/game.py` — règles du jeu (mouvements, barricades, saut, victoire). Pur Python, sans dépendance.
- `barricade/engine/` — IA. Interface `Engine.choose_move(game)`. Placeholders actuels : `random`, `greedy` (distance à l'arrivée).
- `barricade/web/` — serveur local + API JSON + UI HTML/Canvas.
- `tests/` — tests unitaires.

## Feuille de route IA

1. Alpha-beta (negamax) avec ordre des coups + table de transposition.
2. Évaluation : compteur de distance à l'arrivée → différence avec l'adversaire, ratio de confiance pondéré par les barricades adverses.
3. Accélération du hot-loop (numba/C++) si nécessaire.
4. Réseau de neurones (supervisé ou RL) si besoin.