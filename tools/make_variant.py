"""Generate numba-compiled kernel variants with a different static eval.

Reads `barricade/engine/kernel.py`, replaces the body of `eval_fn` with one of
the named variants defined here, and writes a self-contained module into
`barricade/engine/_variants/kernel_<name>.py` (git-ignored). Each variant is
imported with importlib and compiles+caches like the base kernel, so A/B tests
run at full speed.

Variant evals are expressed as the full source of a njit `eval_fn(st)` block.
"""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "barricade" / "engine" / "kernel.py"
OUT_DIR = ROOT / "barricade" / "engine" / "_variants"

_EVAL_START = "@njit(cache=True, inline='always', nogil=True)\ndef eval_fn(st):"

# Match from the eval decorator up to the next "@njit" (start of apply_move).
_BLOCK = re.compile(
    re.escape(_EVAL_START) + r".*?(?=\n@njit)", re.DOTALL
)

_HEADER = (
    '"""Generated kernel variant (eval: {name}). Do not edit; '
    "rebuilt by tools/make_variant.py.\"\"\"\n"
)


def _eval_source(name: str) -> str:
    return VARIANTS[name]


def build(name: str, force: bool = False) -> Path:
    """Write (if missing or force) kernel_<name>.py and return its path."""
    source = KERNEL.read_text(encoding="utf-8")
    block = _BLOCK.search(source)
    if not block:
        raise RuntimeError("eval_fn block not found in kernel.py")
    constants = "\n\n" + _constants(name) + "\n"
    variant = (
        source[: block.start()]
        + _eval_source(name)
        + constants
        + source[block.end():]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"kernel_{name}.py"
    if force or not out.exists():
        out.write_text(_HEADER.format(name=name) + variant, encoding="utf-8")
    return out


def _constants(name: str) -> str:
    consts = ["TEMPO = np.int64(2)"]
    if name == "contact":
        consts.append("CONTACT_W = np.int64(6)")
    if name == "detour":
        consts.append("DETOUR_W = np.int64(2)")
    return "\n".join(consts)


def load(name: str, force: bool = False):
    """Build if needed and import the variant kernel module."""
    path = build(name, force=force)
    mod_name = f"_variants.kernel_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- candidate eval functions -------------------------------------------------

VARIANTS = {}


def _reg(name, doc, body):
    VARIANTS[name] = (
        _EVAL_START
        + "\n"
        + '    """' + doc + '"""\n'
        + body
        + "\n"
    )


_reg(
    "fix1c",
    "baseline: distance advantage, confidence in opponent walls, wall parity.",
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    return dist_adv + (my_w - opp_w)
""",
)


_reg(
    "tempo",
    "fix1c + tempo for the side to move (race games favor acting first).",
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    return dist_adv + (my_w - opp_w) + TEMPO
""",
)


_reg(
    "contact",
    "fix1c + tempo + bonus for pawn contact (adjacent pawns enable jumps/tactics).",
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
        my_p, opp_p = pos0, pos1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
        my_p, opp_p = pos1, pos0
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    mr = my_p // 9
    mc = my_p % 9
    or_ = opp_p // 9
    oc = opp_p % 9
    if abs(mr - or_) <= 1 and abs(mc - oc) <= 1:
        contact = CONTACT_W
    else:
        contact = 0
    return dist_adv + (my_w - opp_w) + TEMPO + contact
""",
)


_reg(
    "detour",
    "fix1c + tempo + obstruction: reward who is actually more obstructed by walls, "
    "measured by (actual BFS distance - manhattan ideal) for each side.",
    """
    pos0, pos1, wl0, wl1, hb_hi, hb_lo, vb_hi, vb_lo, hs_hi, hs_lo, vs_hi, vs_lo, turn, plies, key = st
    d0 = flood_dist(pos0, hb_hi, hb_lo, vb_hi, vb_lo, GOAL0_HI, GOAL0_LO)
    d1 = flood_dist(pos1, hb_hi, hb_lo, vb_hi, vb_lo, GOAL1_HI, GOAL1_LO)
    if turn == 0:
        my_d, opp_d, my_w, opp_w = d0, d1, wl0, wl1
        my_p, opp_p = pos0, pos1
    else:
        my_d, opp_d, my_w, opp_w = d1, d0, wl1, wl0
        my_p, opp_p = pos1, pos0
    m_my = (my_p // 9 if turn == 0 else 8 - my_p // 9) + abs(my_p % 9 - 4)
    m_opp = (opp_p // 9 if turn != 0 else 8 - opp_p // 9) + abs(opp_p % 9 - 4)
    dist_adv = (opp_d - my_d) * DIST_W
    if dist_adv > 0:
        conf = max(1.0 - CONF_W * opp_w, CONF_FLOOR)
        dist_adv = np.int64(dist_adv * conf)
    obstruction = (opp_d - m_opp) - (my_d - m_my)
    return dist_adv + (my_w - opp_w) + TEMPO + DETOUR_W * obstruction
""",
)