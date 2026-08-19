"""Local web server + JSON API for the Barricade UI."""

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from urllib.parse import parse_qs, urlparse

from barricade.game import Barricade
from barricade.engine.alphabeta import AlphaBetaEngine
from barricade.engine.greedy import GreedyEngine
from barricade.engine.random import RandomEngine
from barricade.engine import evaluate

try:
    from barricade.engine.numba_engine import KernelEngine
    _HAS_NUMBA = True
except ImportError:
    KernelEngine = None
    _HAS_NUMBA = False

try:
    from barricade.engine.mcts import MctsEngine
    _HAS_MCTS = True
except ImportError:
    MctsEngine = None
    _HAS_MCTS = False

try:
    from barricade.engine.analysis import AnalysisSession
    _HAS_ANALYSIS = True
except ImportError:
    AnalysisSession = None
    _HAS_ANALYSIS = False

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ENGINES = {"random": RandomEngine, "greedy": GreedyEngine}
if _HAS_NUMBA:
    ENGINES["alphabeta"] = KernelEngine
    ENGINES["kernel"] = KernelEngine
else:
    ENGINES["alphabeta"] = AlphaBetaEngine
if _HAS_MCTS:
    ENGINES["mcts"] = MctsEngine
DEFAULT_ENGINE = "kernel" if _HAS_NUMBA else "alphabeta"

MAX_SNAPSHOTS = 600
MATCH_MAX_PLIES = 400


class App:
    def __init__(self, depth=8, time_limit=2.0):
        self.lock = RLock()
        self.game = Barricade()
        self.ai_player = None  # None=pvp, 0/1=vs AI, 'both'=AI vs AI
        self.mode = "pvp"
        self.engine_name = None
        self.engine_info = None
        self.thinking = False
        self.depth = depth
        self.time_limit = time_limit
        self.engine = self._make_engine(DEFAULT_ENGINE)
        self.analysis = None
        self.snapshots = []
        self.match = None
        self._match_thread = None
        self._match_stop = threading.Event()
        self._push_snapshot()

    def _with_lock(self, fn):
        with self.lock:
            return fn()

    def meta(self):
        return {"engines": sorted(ENGINES), "default": DEFAULT_ENGINE}

    def _make_engine(self, name, time_limit=None, depth=None):
        cls = ENGINES.get(name, ENGINES[DEFAULT_ENGINE])
        tl = self.time_limit if time_limit is None else time_limit
        dp = self.depth if depth is None else depth
        if cls in (AlphaBetaEngine, KernelEngine):
            return cls(max_depth=dp, time_limit=tl)
        if cls is MctsEngine:
            return cls(time_limit=tl)
        return cls()

    def _snapshot_of(self, game):
        d = game.to_dict()
        d["ai_player"] = self.ai_player
        d["mode"] = self.mode
        d["engine"] = self.engine_name
        d["engine_info"] = self.engine_info
        return d

    def _snapshot(self):
        return self._snapshot_of(self.game)

    def _push_snapshot(self):
        self.snapshots.append(self._snapshot())
        if len(self.snapshots) > MAX_SNAPSHOTS:
            self.snapshots = self.snapshots[-MAX_SNAPSHOTS:]

    def _stop_analysis_locked(self):
        if self.analysis is not None:
            self.analysis.stop()
            self.analysis = None

    def new(self, mode, ai_player=1, engine="alphabeta", time_control=None):
        def _new():
            if self.match_running():
                return {"error": "match in progress"}
            self.game = Barricade(time_control=time_control)
            self.mode = mode
            self.engine_info = None
            self._stop_analysis_locked()
            if mode == "ai":
                self.ai_player = ai_player if ai_player in (0, 1) else 1
                self.engine = self._make_engine(engine)
                self.engine_name = engine
            elif mode == "ai2":
                self.ai_player = "both"
                self.engine = self._make_engine(engine)
                self.engine_name = engine
            else:
                self.ai_player = None
                self.engine_name = None
            self.snapshots = []
            self._push_snapshot()
            return self.state()
        return self._with_lock(_new)

    def state(self):
        def _state():
            d = self.game.to_dict()
            d["ai_player"] = self.ai_player
            d["mode"] = self.mode
            d["engine"] = self.engine_name
            d["engine_info"] = self.engine_info
            return d
        return self._with_lock(_state)

    def play(self, move):
        def _play():
            if self.match_running():
                return None, "match in progress"
            try:
                if self.thinking:
                    return None, "AI thinking"
                if move[0] == "move":
                    self.game.apply(("move", tuple(move[1])))
                elif move[0] == "wall":
                    self.game.apply(("wall", (move[1][0], move[1][1], move[1][2])))
                else:
                    return None, "bad action"
                self._stop_analysis_locked()
                self._push_snapshot()
                return self.state(), None
            except ValueError as e:
                return None, str(e)
        return self._with_lock(_play)

    def ai_move(self):
        def _ai():
            with self.lock:
                if self.match_running():
                    return None, "match in progress"
                if self.ai_player is None:
                    return None, "AI not active"
                if self.game.winner is not None:
                    return None, "game over"
                if self.ai_player != "both" and self.game.turn != self.ai_player:
                    return None, "not AI turn"
                if self.thinking:
                    return None, "AI busy"
                self.thinking = True
                self.engine_info = None
                self._set_engine_time(self.game.turn)
            try:
                action = self.engine.choose_move(self.game)
            finally:
                with self.lock:
                    self.thinking = False
            if action is None:
                return None, "no move"
            with self.lock:
                try:
                    self.game.apply(action)
                except ValueError as e:
                    return None, str(e)
                info = getattr(self.engine, "last_info", None)
                self.engine_info = {"engine": self.engine_name, "info": info}
                self._stop_analysis_locked()
                self._push_snapshot()
            return self.state(), None
        return _ai()

    def info(self):
        """Cheap, lock-free: live search progress while the AI thinks."""
        return {"thinking": self.thinking,
                "engine": self.engine_name,
                "progress": getattr(self.engine, "progress", None)}

    def analysis_start(self, engine="kernel", max_depth=None, mcts_nodes=None, mcts_tick=None):
        if not _HAS_ANALYSIS:
            return {"error": "analysis requires numba"}
        if engine not in ("kernel", "mcts"):
            engine = "kernel"
        self._stop_analysis_locked()
        self.analysis = AnalysisSession(
            self.game, engine=engine,
            max_depth=max_depth, mcts_nodes=mcts_nodes, mcts_tick=mcts_tick)
        self.analysis.start()
        return self.analysis.state()

    def analysis_stop(self):
        def _stop():
            if self.analysis is not None:
                self.analysis.stop()
                state = self.analysis.state()
                self.analysis = None
                return state
            return {"running": False}
        return self._with_lock(_stop)

    def analysis_state(self):
        if self.analysis is None:
            return {"running": False}
        return self.analysis.state()

    def eval_info(self):
        def _eval():
            g = self.game
            d0 = g.dist_to_goal(0)
            d1 = g.dist_to_goal(1)
            w0 = g.walls_left[0]
            w1 = g.walls_left[1]
            p = g.turn
            my_d, opp_d, my_w, opp_w = (d0, d1, w0, w1) if p == 0 else (d1, d0, w1, w0)
            raw_gap = (opp_d - my_d) * evaluate.DIST_W
            conf = 1.0
            if raw_gap > 0:
                conf = max(1.0 - evaluate.CONF_W * opp_w, evaluate.CONF_FLOOR)
            gap_adj = int(raw_gap * conf) if raw_gap > 0 else int(raw_gap)
            score = gap_adj + (my_w - opp_w)
            gap_sq = opp_d - my_d
            win = 0.5 + 0.5 * math.tanh(gap_sq / 4.0)
            return {
                "turn": p,
                "d0": d0, "d1": d1,
                "walls0": w0, "walls1": w1,
                "dist_gap": gap_sq,
                "confidence": round(conf, 3),
                "score": int(score),
                "win_chance": round(win, 4),
                "components": [
                    {"name": "Avance en cases",
                     "formula": f"({opp_d} − {my_d}) × 4 = {int(raw_gap)}",
                     "note": "distance adverse − distance perso"},
                    {"name": "Confiance",
                     "formula": f"max(1 − 0.12 × {opp_w}, 0.3) = {conf:.2f}",
                     "note": "appliquée seulement quand on est en avance"},
                    {"name": "Parité de murs",
                     "formula": f"{my_w} − {opp_w} = {my_w - opp_w}",
                     "note": "murs restants − murs adverses"},
                    {"name": "Score total",
                     "formula": f"{gap_adj} + ({my_w - opp_w}) = {int(score)}",
                     "note": "perspective du joueur à jouer"},
                ],
            }
        return self._with_lock(_eval)

    def history(self):
        def _h():
            return {"count": len(self.snapshots), "snapshots": self.snapshots}
        return self._with_lock(_h)

    def snapshot(self, ply):
        def _s():
            if not self.snapshots:
                return None
            i = max(0, min(int(ply), len(self.snapshots) - 1))
            return self.snapshots[i]
        return self._with_lock(_s)

    # ----- bot vs bot matches -----

    def match_running(self):
        return self.match is not None and self.match.get("running") is True

    def start_match(self, p1, p2, time_limit=1.0, games=1, depth=10):
        games = max(1, min(int(games), 20))
        time_limit = max(0.05, float(time_limit))
        depth = max(1, min(int(depth), 30))

        def _start():
            if self.match_running():
                return None, "match already running"
            for e in (p1, p2):
                if e not in ENGINES:
                    return None, f"unknown engine {e}"
            self.match = {
                "running": True, "p1": p1, "p2": p2,
                "time_limit": time_limit, "depth": depth, "games": games,
                "current": 0, "ply": 0, "moves": [], "results": [],
                "score": [0, 0], "draws": 0, "status": "starting",
                "error": None,
            }
            self._match_stop = threading.Event()
            self._match_thread = threading.Thread(
                target=self._run_match, args=(p1, p2, time_limit, games, depth), daemon=True)
            self._match_thread.start()
            return dict(self.match), None
        return self._with_lock(_start)

    def _run_match(self, p1, p2, time_limit, games, depth):
        ea = self._make_engine(p1, time_limit=time_limit, depth=depth)
        eb = self._make_engine(p2, time_limit=time_limit, depth=depth)
        last_game = None
        try:
            for gi in range(games):
                g = Barricade()
                with self.lock:
                    self.match["current"] = gi + 1
                    self.match["ply"] = 0
                    self.match["status"] = f"partie {gi + 1}/{games}"
                ply = 0
                while g.winner is None and ply < MATCH_MAX_PLIES and not self._match_stop.is_set():
                    eng = ea if g.turn == 0 else eb
                    with self.lock:
                        self.match["moves"] = [m[2] for m in g.history]
                    try:
                        action = eng.choose_move(g)
                    except Exception:
                        action = None
                    if action is None:
                        g.winner = 1 - g.turn
                        g.game_over_reason = "resign"
                        break
                    try:
                        g.apply(action)
                    except ValueError:
                        g.winner = 1 - g.turn
                        g.game_over_reason = "illegal"
                        break
                    ply += 1
                    with self.lock:
                        self.match["ply"] = ply
                if self._match_stop.is_set():
                    break
                with self.lock:
                    self.match["results"].append({
                        "game": gi + 1,
                        "winner": g.winner,
                        "plies": len(g.history),
                        "reason": g.game_over_reason,
                    })
                    if g.winner == 0:
                        self.match["score"][0] += 1
                    elif g.winner == 1:
                        self.match["score"][1] += 1
                    else:
                        self.match["draws"] += 1
                    self.match["moves"] = [m[2] for m in g.history]
                last_game = g
            with self.lock:
                self.match["running"] = False
                self.match["status"] = "done"
                self.match["error"] = None
                if last_game is not None:
                    self._load_game_for_review(last_game)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.match["running"] = False
                self.match["status"] = "error"
                self.match["error"] = str(e)
        finally:
            with self.lock:
                self.match["running"] = False

    def stop_match(self):
        def _stop():
            if not self.match_running():
                return None
            self._match_stop.set()
            return dict(self.match)
        return self._with_lock(_stop)

    def match_state(self):
        def _s():
            if self.match is None:
                return {"running": False}
            return dict(self.match)
        return self._with_lock(_s)

    def _load_game_for_review(self, game):
        self.game = game
        self.ai_player = "both"
        self.mode = "ai2"
        self.engine_name = "match"
        self.engine = None
        self.engine_info = None
        self._stop_analysis_locked()
        self.snapshots = []
        g2 = Barricade()
        self.snapshots.append(self._snapshot_of(g2))
        for m in game.history:
            g2.apply((m[0], m[2]), check=False)
            self.snapshots.append(self._snapshot_of(g2))
        self.game = g2


class Handler(BaseHTTPRequestHandler):
    server_version = "Barricade/1.0"
    app = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, name, ctype):
        path = os.path.join(STATIC_DIR, name)
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        if path == "/":
            self._file("index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._file("style.css", "text/css; charset=utf-8")
        elif path == "/main.js":
            self._file("main.js", "text/javascript; charset=utf-8")
        elif path == "/api/state":
            self._json(Handler.app.state())
        elif path == "/api/meta":
            self._json(Handler.app.meta())
        elif path == "/api/info":
            self._json(Handler.app.info())
        elif path == "/api/analysis":
            self._json(Handler.app.analysis_state())
        elif path == "/api/history":
            self._json(Handler.app.history())
        elif path == "/api/snapshot":
            ply = int(query.get("ply", ["0"])[0])
            self._json(Handler.app.snapshot(ply))
        elif path == "/api/eval":
            self._json(Handler.app.eval_info())
        elif path == "/api/match/state":
            self._json(Handler.app.match_state())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        app = Handler.app
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if path == "/api/new":
            self._json(app.new(
                body.get("mode", "pvp"),
                body.get("ai_player", 1),
                body.get("engine", "greedy"),
                body.get("time_control"),
            ))
        elif path == "/api/move":
            res, err = app.play(body.get("move"))
            self._json(res, 200 if err is None else 400)
        elif path == "/api/ai":
            res, err = app.ai_move()
            self._json(res, 200 if err is None else 400)
        elif path == "/api/analysis/start":
            self._json(app.analysis_start(
                body.get("engine", "kernel"),
                body.get("max_depth"),
                body.get("mcts_nodes"),
                body.get("mcts_tick"),
            ))
        elif path == "/api/analysis/stop":
            self._json(app.analysis_stop())
        elif path == "/api/match/start":
            res, err = app.start_match(
                body.get("p1", "kernel"),
                body.get("p2", "mcts"),
                body.get("time", 1.0),
                body.get("games", 1),
                body.get("depth", 10),
            )
            self._json(res, 200 if err is None else 400)
        elif path == "/api/match/stop":
            self._json(app.stop_match())
        else:
            self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8000, depth=8, time_limit=2.0):
    app = App(depth=depth, time_limit=time_limit)
    Handler.app = app
    if hasattr(app.engine, "_warmup"):
        app.engine._warmup()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Barricade on http://{host}:{port}  (AI: depth {depth}, {time_limit}s)")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()