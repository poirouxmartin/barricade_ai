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

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
ENGINES = {"random": RandomEngine, "greedy": GreedyEngine, "alphabeta": AlphaBetaEngine}


class App:
    def __init__(self, depth=8, time_limit=2.0):
        self.lock = RLock()
        self.game = Barricade()
        self.ai_player = None  # None=pvp, 0/1=vs AI, 'both'=AI vs AI
        self.depth = depth
        self.time_limit = time_limit
        self.engine = self._make_engine("alphabeta")

    def _with_lock(self, fn):
        with self.lock:
            return fn()

    def new(self, mode, ai_player=1, engine="alphabeta"):
        def _new():
            self.game = Barricade()
            if mode == "ai":
                self.ai_player = ai_player if ai_player in (0, 1) else 1
                self.engine = self._make_engine(engine)
            elif mode == "ai2":
                self.ai_player = "both"
                self.engine = self._make_engine(engine)
            else:
                self.ai_player = None
            return self.state()
        return self._with_lock(_new)

    def _make_engine(self, name):
        cls = ENGINES.get(name, AlphaBetaEngine)
        if cls is AlphaBetaEngine:
            return cls(max_depth=self.depth, time_limit=self.time_limit)
        return cls()

    def state(self):
        def _state():
            d = self.game.to_dict()
            d["ai_player"] = self.ai_player
            return d
        return self._with_lock(_state)

    def play(self, move):
        def _play():
            try:
                if move[0] == "move":
                    self.game.apply(("move", tuple(move[1])))
                elif move[0] == "wall":
                    self.game.apply(("wall", (move[1][0], move[1][1], move[1][2])))
                else:
                    return None, "bad action"
                self._auto_ai()
                return self.state(), None
            except ValueError as e:
                return None, str(e)
        return self._with_lock(_play)

    def ai_move(self):
        def _ai():
            if self.ai_player is None or self.ai_player == "both":
                return None, "AI not active"
            if self.game.winner is not None or self.game.turn != self.ai_player:
                return None, "not AI turn"
            self.game.apply(self.engine.choose_move(self.game))
            return self.state(), None
        return self._with_lock(_ai)

    def _auto_ai(self):
        if self.ai_player is None:
            return
        if self.game.winner is not None:
            return
        if self.ai_player == "both" or self.game.turn == self.ai_player:
            self.game.apply(self.engine.choose_move(self.game))
            self._auto_ai()  # in 'both' mode keep going until a human turn or game over


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
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        app = Handler.app
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if path == "/api/new":
            self._json(app.new(body.get("mode", "pvp"), body.get("ai_player", 1), body.get("engine", "greedy")))
        elif path == "/api/move":
            res, err = app.play(body.get("move"))
            self._json(res, 200 if err is None else 400)
        elif path == "/api/ai":
            res, err = app.ai_move()
            self._json(res, 200 if err is None else 400)
        else:
            self._json({"error": "not found"}, 404)


def serve(host="127.0.0.1", port=8000, depth=8, time_limit=2.0):
    Handler.app = App(depth=depth, time_limit=time_limit)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Barricade on http://{host}:{port}  (AI: depth {depth}, {time_limit}s)")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()