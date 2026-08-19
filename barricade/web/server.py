"""Local web server + JSON API for the Barricade UI."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from urllib.parse import urlparse

from barricade.game import Barricade
from barricade.engine.alphabeta import AlphaBetaEngine
from barricade.engine.greedy import GreedyEngine
from barricade.engine.random import RandomEngine

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

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ENGINES = {"random": RandomEngine, "greedy": GreedyEngine}
if _HAS_NUMBA:
    # alpha-beta is backed by the compiled kernel when numba is available
    ENGINES["alphabeta"] = KernelEngine
    ENGINES["kernel"] = KernelEngine
else:
    ENGINES["alphabeta"] = AlphaBetaEngine
if _HAS_MCTS:
    ENGINES["mcts"] = MctsEngine
DEFAULT_ENGINE = "kernel" if _HAS_NUMBA else "alphabeta"


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

    def _with_lock(self, fn):
        with self.lock:
            return fn()

    def meta(self):
        return {"engines": sorted(ENGINES), "default": DEFAULT_ENGINE}

    def new(self, mode, ai_player=1, engine="alphabeta", time_control=None):
        def _new():
            self.game = Barricade(time_control=time_control)
            self.mode = mode
            self.engine_info = None
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
            return self.state()
        return self._with_lock(_new)

    def _make_engine(self, name):
        cls = ENGINES.get(name, ENGINES[DEFAULT_ENGINE])
        if cls in (AlphaBetaEngine, KernelEngine):
            return cls(max_depth=self.depth, time_limit=self.time_limit)
        if cls is MctsEngine:
            return cls(time_limit=self.time_limit)
        return cls()

    def _set_engine_time(self, player):
        """Clamp the engine's time budget to the player's remaining clock."""
        if not self.game.time_left:
            return
        left = self.game.time_left[player]
        if left is None:
            return
        budget = max(0.05, min(self.time_limit, left - 0.2))
        if hasattr(self.engine, "time_limit"):
            self.engine.time_limit = budget

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
            try:
                if self.thinking:
                    return None, "AI thinking"
                if move[0] == "move":
                    self.game.apply(("move", tuple(move[1])))
                elif move[0] == "wall":
                    self.game.apply(("wall", (move[1][0], move[1][1], move[1][2])))
                else:
                    return None, "bad action"
                return self.state(), None
            except ValueError as e:
                return None, str(e)
        return self._with_lock(_play)

    def ai_move(self):
        def _ai():
            with self.lock:
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
                # search runs without the lock so /api/state and /api/info stay responsive
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
            return self.state(), None
        return _ai()

    def info(self):
        """Cheap, lock-free: live search progress while the AI thinks."""
        return {"thinking": self.thinking,
                "engine": self.engine_name,
                "progress": getattr(self.engine, "progress", None)}


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
        if path == "/" or path == "/index.html":
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
        else:
            self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8000, depth=8, time_limit=2.0):
    app = App(depth=depth, time_limit=time_limit)
    Handler.app = app
    # compile numba once at startup so the first AI move is not slowed down
    if hasattr(app.engine, "_warmup"):
        app.engine._warmup()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Barricade on http://{host}:{port}  (AI: depth {depth}, {time_limit}s)")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()