"""Train the MCTS policy net (move priors) and export its weights.

Behavior cloning: a tanh MLP NN_IN -> NN_HID -> NN_HID -> NN_POL maps the
board from the mover's perspective to raw logits over the 209 actions, trained
with cross-entropy on the moves played by the kernel engines in the self-play
games (tools/selfplay_policy.py). The weights are consumed by
`barricade.engine.kernel.nn_policy_logits`.

Usage:
    python tools/train_policy.py --data data/policy.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.head(x)


def topk_acc(logits, targets, k):
    top = torch.topk(logits, k, dim=1).indices
    hit = (top == targets.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/policy.npz")
    ap.add_argument("--out", default="barricade/engine/policy_weights.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data)
    X = torch.tensor(d["X"], dtype=torch.float32)
    A = torch.tensor(d["A"], dtype=torch.long)
    n = len(X)
    n_val = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xv, Av = X[val_idx], A[val_idx]
    Xt, At = X[tr_idx], A[tr_idx]

    model = PolicyNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = None
    wait = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), At[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), Av).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  epoch {ep + 1}: val loss {vloss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t1 = topk_acc(model(Xt), At, 1)
        t5 = topk_acc(model(Xt), At, 5)
        v1 = topk_acc(model(Xv), Av, 1)
        v5 = topk_acc(model(Xv), Av, 5)
    print(f"train top-1 {t1:.3f} / top-5 {t5:.3f}")
    print(f"val   top-1 {v1:.3f} / top-5 {v5:.3f}   best val loss {best_val:.4f}")

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

    # sanity: numba nn_policy_logits must match torch forward. Reconstruct a
    # state from each val feature row (pawn = argmax of one-hot, walls = set
    # slots, walls left = raw) and compare the logits.
    feats = d["X"][val_idx.numpy()]
    with torch.no_grad():
        torch_logits = model(torch.tensor(feats, dtype=torch.float32)).numpy()
    nb_logits = np.zeros((len(feats), POL))
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
        kernel.nn_policy_logits(st, out)
        nb_logits[i] = out
    diff = np.abs(nb_logits - torch_logits)
    print(f"numba vs torch: max |diff| {diff.max():.6f} on {len(feats)} val rows")


if __name__ == "__main__":
    main()