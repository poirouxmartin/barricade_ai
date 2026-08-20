"""Train the MCTS policy + value net and export its weights.

Two-head MLP: a tanh trunk NN_IN -> NN_HID -> NN_HID feeds a policy head
(NN_POL logits) and a value head (tanh, win prob of the side to move). Trained
with cross-entropy on the moves the kernel engines played and MSE on the game
outcome (tools/selfplay_policy.py). Consumed by
`barricade.engine.kernel.nn_policy_value`.

Usage:
    python tools/train_policy.py --data data/policy.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barricade.engine import kernel

IN = kernel.NN_IN
HID = kernel.NN_HID
POL = kernel.NN_POL


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, POL)
        self.fv = nn.Linear(HID, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x), torch.tanh(self.fv(x))


def topk_acc(logits, targets, k):
    top = torch.topk(logits, k, dim=1).indices
    hit = (top == targets.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean())


def sign_acc(pred, y):
    agree = (pred > 0) == (y > 0)
    return float(agree.float().mean())


def row_weights(X):
    """Per-row loss weight: near-terminal rows (mover within 2 plies of its
    goal) are rare and carry the conversion signal, so upweight them."""
    my_dist = X[:, IN - 2] * 16.0
    w = torch.ones(len(X), dtype=torch.float32)
    w[my_dist <= 2.0] += 5.0
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/policy.npz")
    ap.add_argument("--out", default="barricade/engine/policy_weights.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--value-weight", type=float, default=0.0,
                    help="weight of the value-head MSE; 0 trains policy only "
                         "(the value head is an experimental leaf evaluator "
                         "and degrades the policy when trained jointly)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data)
    X = torch.tensor(d["X"], dtype=torch.float32)
    A = torch.tensor(d["A"], dtype=torch.long)
    V = torch.tensor(d["V"], dtype=torch.float32)
    n = len(X)
    n_val = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xv, Av, Vv = X[val_idx], A[val_idx], V[val_idx]
    Xt, At, Vt = X[tr_idx], A[tr_idx], V[tr_idx]
    w = row_weights(X)
    Wv, Wt = w[val_idx], w[tr_idx]

    model = PolicyNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None
    wait = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            lp, vp = model(Xt[idx])
            ce = (F.cross_entropy(lp, At[idx], reduction="none") * Wt[idx]).mean()
            mse = (F.mse_loss(vp.squeeze(-1), Vt[idx], reduction="none")
                   * Wt[idx]).mean()
            loss = ce + args.value_weight * mse
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lp, vp = model(Xv)
            ce = (F.cross_entropy(lp, Av, reduction="none") * Wv).mean()
            mse = (F.mse_loss(vp.squeeze(-1), Vv, reduction="none")
                   * Wv).mean()
            vloss = ce + args.value_weight * mse
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  epoch {ep + 1}: val loss {vloss.item():.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lp, vp = model(Xt)
        t1 = topk_acc(lp, At, 1)
        t5 = topk_acc(lp, At, 5)
        tv = sign_acc(vp.squeeze(-1), Vt)
        lp, vp = model(Xv)
        v1 = topk_acc(lp, Av, 1)
        v5 = topk_acc(lp, Av, 5)
        vv = sign_acc(vp.squeeze(-1), Vv)
    print(f"train top-1 {t1:.3f} / top-5 {t5:.3f}   value sign acc {tv:.3f}")
    print(f"val   top-1 {v1:.3f} / top-5 {v5:.3f}   value sign acc {vv:.3f}   "
          f"best val loss {best_val:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = {
        "w1": model.fc1.weight.detach().numpy(),
        "b1": model.fc1.bias.detach().numpy(),
        "w2": model.fc2.weight.detach().numpy(),
        "b2": model.fc2.bias.detach().numpy(),
        "wp": model.head.weight.detach().numpy(),
        "bp": model.head.bias.detach().numpy(),
    }
    if args.value_weight > 0:
        w["wv"] = model.fv.weight.detach().numpy()
        w["bv"] = model.fv.bias.detach().numpy()
    np.savez(out, **w)
    print(f"saved weights -> {out}")

    # invalidate numba's disk cache (weights are module globals captured at
    # compile time) and re-import so the kernel picks up the fresh values
    import shutil

    for cache_dir in Path("barricade").rglob("__pycache__"):
        for f in list(cache_dir.glob("kernel*.nbc")) + list(cache_dir.glob("kernel*.nbi")):
            f.unlink()
    print("numba kernel cache purged")

    import importlib

    sys.modules.pop("barricade.engine.kernel", None)
    kernel = importlib.import_module("barricade.engine.kernel")

    # sanity: numba nn_policy_value must match torch forward. Reconstruct a
    # state from each val feature row (pawn = argmax of one-hot, walls = set
    # slots, walls left = raw) and compare logits and value.
    feats = d["X"][val_idx.numpy()]
    with torch.no_grad():
        torch_logits, torch_value = model(torch.tensor(feats, dtype=torch.float32))
        torch_logits = torch_logits.numpy()
        torch_value = torch_value.numpy()[:, 0]
    nb_logits = np.zeros((len(feats), POL))
    nb_value = np.zeros(len(feats))
    for i in range(len(feats)):
        f = feats[i]
        my = int(np.argmax(f[:81]))
        opp = int(np.argmax(f[81:162]))
        hs = 0
        vs = 0
        for k in range(64):
            if f[162 + k] > 0.5:
                hs |= 1 << k
            if f[226 + k] > 0.5:
                vs |= 1 << k
        st = (np.int64(my), np.int64(opp), np.int64(round(f[290] * 10)),
              np.int64(round(f[291] * 10)), np.uint64(0), np.uint64(0),
              np.uint64(0), np.uint64(0), np.uint64(hs), np.uint64(0),
              np.uint64(vs), np.uint64(0), np.int64(0), np.int64(0),
              np.int64(0))
        out = np.zeros(POL, np.float64)
        val = np.zeros(1, np.float64)
        kernel.nn_policy_value(st, f[IN - 2] * 16, f[IN - 1] * 16, out, val)
        nb_logits[i] = out
        nb_value[i] = val[0]
    dl = np.abs(nb_logits - torch_logits).max()
    print(f"numba vs torch: max |diff| logits {dl:.6f} "
          f"on {len(feats)} val rows")
    if args.value_weight > 0:
        dv = np.abs(nb_value - torch_value).max()
        print(f"numba vs torch: max |diff| value {dv:.6f}")


if __name__ == "__main__":
    main()