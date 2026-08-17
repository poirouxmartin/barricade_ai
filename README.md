# Barricade

Jeu de stratégie 2 joueurs sur grille 9×9 (type Quoridor). Course vers le côté opposé du plateau, 10 barricades chacun, saut par-dessus l'adversaire.

## Lancer

```bash
python run.py                    # http://127.0.0.1:8000
python run.py --depth 6 --time 1 # AI params (defaut: depth 8, 2s)
python -m unittest discover tests -v   # tests
```

Modes via "New game" : PvP, vs AI (greedy), vs AI (alpha-beta), AI vs AI.

## Structure

- `barricade/game.py` — règles du jeu (mouvements, barricades, saut, victoire). Pur Python, sans dépendance.
- `barricade/engine/` — IA. Interface `Engine.choose_move(game)`.
  - `alphabeta.py` — negamax alpha-beta : deepening itératif, table de transposition, murs ciblés sur le chemin adverse.
  - `evaluate.py` — évaluation : distance à l'arrivée (différence), ratio de confiance pondéré par les barricades adverses, avantage de murs restants.
  - `greedy.py`, `random.py` — placeholders.
- `barricade/web/` — serveur local + API JSON + UI HTML/Canvas.
- `tests/` — tests unitaires.

## Feuille de route IA

1. ✅ Alpha-beta (negamax) : deepening itératif, table de transposition, ordre des coups, murs ciblés.
2. ✅ Évaluation : distance à l'arrivée → différence avec l'adversaire, ratio de confiance pondéré par les barricades adverses.
3. ⏳ Accélération du hot-loop (numba/C++/bitboards) pour gagner en profondeur.
4. ⏳ Réseau de neurones (supervisé ou RL) si besoin.