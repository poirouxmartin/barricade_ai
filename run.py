"""Launch the Barricade web UI.

Usage: python run.py [--host 127.0.0.1] [--port 8000]
"""

import argparse

from barricade.web.server import serve


def main():
    p = argparse.ArgumentParser(description="Barricade local server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--depth", type=int, default=8, help="max kernel/alpha-beta depth")
    p.add_argument("--time", type=float, default=2.0, help="kernel/alpha-beta time limit (s)")
    args = p.parse_args()
    serve(args.host, args.port, args.depth, args.time)


if __name__ == "__main__":
    main()