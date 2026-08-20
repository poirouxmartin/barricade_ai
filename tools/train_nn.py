"""Train the small MLP for the learned eval and export its weights.

Reads a self-play dataset (tools/selfplay.py), trains a tanh MLP
11 -> 32 -> 32 -> 1 to predict the game outcome from P1's perspective
(+1 P1 wins / -1 P2 wins / 0 draw), and writes the weights as a numpy npz
consumed by `barricade.engine.kernel.nn_value`.

Usage:
    python tools/train_nn.py --data data/positions.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from barricade.engine import kernel

FEATS = kernel.MLP_FEATS
HID = kernel.MLP_HID


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATS, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.fc3 = nn.Linear(HID, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))


def sign_acc(pred, y):
    pred = pred.squeeze(-1)
    agree = (pred > 0) == (y > 0)
    exact = agree & (y != 0)
    return float(exact.float().mean()), float(agree.float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/positions.npz")
    ap.add_argument("--out", default="barricade/engine/nn_weights.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    n = len(X)
    n_val = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xv, yv = X[val_idx], y[val_idx]
    Xt, yt = X[tr_idx], y[tr_idx]

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    wait = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv).squeeze(-1)
            vloss = loss_fn(pred, yv).item()
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
        tv, ta = sign_acc(model(Xt), yt)
        vv, va = sign_acc(model(Xv), yv)
    print(f"train sign acc {ta:.3f} (exact {tv:.3f})")
    print(f"val   sign acc {va:.3f} (exact {vv:.3f})   best val loss {best_val:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = {
        "w1": model.fc1.weight.detach().numpy(),
        "b1": model.fc1.bias.detach().numpy(),
        "w2": model.fc2.weight.detach().numpy(),
        "b2": model.fc2.bias.detach().numpy(),
        "w3": model.fc3.weight.detach().numpy(),
        "b3": model.fc3.bias.detach().numpy(),
    }
    np.savez(out, **w)
    print(f"saved weights -> {out}")

    # numba's disk cache bakes the weight globals into the compiled functions,
    # so a retrain must invalidate it or fresh processes would keep the stale
    # weights. The re-import below recompiles with the new values.
    import shutil

    for cache_dir in Path("barricade").rglob("__pycache__"):
        for f in list(cache_dir.glob("kernel*.nbc")) + list(cache_dir.glob("kernel*.nbi")):
            f.unlink()
    print("numba kernel cache purged")

    # re-import the kernel so its MLP globals pick up the freshly saved weights
    import importlib

    sys.modules.pop("barricade.engine.kernel", None)
    kernel = importlib.import_module("barricade.engine.kernel")

    # sanity: numba nn_value must match torch forward on exactly-reconstructable
    # rows (those where every feature is below its cap, so the raw inputs can be
    # recovered from the normalized training features).
    feats = d["X"][val_idx.numpy()]
    d0 = feats[:, 0] * 16
    d1 = feats[:, 1] * 16
    pos0 = feats[:, 2] * 8 * 9 + feats[:, 3] * 8
    pos1 = (8 - feats[:, 4] * 8) * 9 + feats[:, 5] * 8
    wl0 = feats[:, 6] * 10
    wl1 = feats[:, 7] * 10
    plies = feats[:, 10] * 40
    re_f8 = np.minimum(d0 - (pos0 // 9 + np.abs(pos0 % 9 - 4)), 16) / 16
    re_f9 = np.minimum(d1 - ((8 - pos1 // 9) + np.abs(pos1 % 9 - 4)), 16) / 16
    ok = (np.abs(re_f8 - feats[:, 8]) < 1e-9) & \
         (np.abs(re_f9 - feats[:, 9]) < 1e-9)
    n_ok = int(ok.sum())
    with torch.no_grad():
        torch_out = model(Xv[ok]).squeeze(-1).numpy()
    nb = np.array([kernel.nn_value(int(d0[i]), int(d1[i]), int(pos0[i]),
                                   int(pos1[i]), int(wl0[i]), int(wl1[i]),
                                   int(plies[i])) for i in np.nonzero(ok)[0]])
    diff = np.abs(nb - torch_out)
    print(f"numba vs torch: max |diff| {diff.max():.6f} on {n_ok}/{len(feats)} "
          f"reconstructable val rows")


if __name__ == "__main__":
    main()