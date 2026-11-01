"""Train an offline velocity estimator on logged rollouts.

Predicts true base linear velocity (vx, vy, vz, body frame) from a short history of the
blind actor's observations (IMU + joint encoders + command + previous action).

Compares against two baselines:
  - mean:     always predict the training-set mean velocity (no information)
  - command:  predict the commanded velocity (vx, vy from the RC command, vz = 0),
              i.e. "assume the robot does exactly what it's told"
and an ablation over history length (1 step vs 5 steps).

Usage (from C:\\rl\\IsaacLab):
  uv run --with matplotlib python scripts/train_velocity_estimator.py --data logs/velocity_data.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--data", default="logs/velocity_data.pt")
p.add_argument("--out_dir", default="logs/velocity_estimator")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)

d = torch.load(args.data)
X_all, Y_all, D_all = d["policy_obs"].float(), d["true_vel"].float(), d["dones"].bool()
T, N, F = X_all.shape
print(f"Loaded T={T} steps x N={N} robots, obs dim {F}")
CMD = slice(6, 9)  # velocity_commands inside the 45-dim policy obs (after ang_vel 0:3, gravity 3:6)

# Split by ROBOT, not by time step, so test robots are never seen during training.
perm = torch.randperm(N)
n_test = max(1, N // 5)
test_envs, train_envs = perm[:n_test], perm[n_test:]


def windows(H: int, envs: torch.Tensor):
    """Stack H-step histories that never cross an episode boundary."""
    X, D, Y = X_all[:, envs], D_all[:, envs], Y_all[:, envs]
    # valid[t] = no episode ended during steps t-H+1 .. t-1  (done[t'] means reset after step t')
    valid = torch.ones(T, X.shape[1], dtype=torch.bool)
    valid[: H - 1] = False
    for k in range(1, H):
        valid[H - 1 :] &= ~D[H - 1 - k : T - k]
    feats = [X[H - 1 - k : T - k] for k in range(H - 1, -1, -1)]  # oldest ... newest
    Xh = torch.cat(feats, dim=-1)  # [T-H+1, n, H*F]
    v = valid[H - 1 :]
    return Xh[v], Y[H - 1 :][v], X[H - 1 :][v][:, CMD]


def rmse(pred, true):
    return torch.sqrt(((pred - true) ** 2).mean(0)).tolist()


class MLP(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 3))

    def forward(self, x):
        return self.net(x)


def train_eval(H: int):
    Xtr, Ytr, _ = windows(H, train_envs)
    Xte, Yte, Cte = windows(H, test_envs)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n, Xte_n = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev)
    Ytr_d, Yte_d = Ytr.to(dev), Yte.to(dev)
    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bs = 4096
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(Xtr_n.shape[0], device=dev)
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            loss = ((model(Xtr_n[b]) - Ytr_d[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                te = ((model(Xte_n) - Yte_d) ** 2).mean().sqrt().item()
            print(f"  H={H} epoch {ep + 1}/{args.epochs}  test RMSE (all axes) {te:.4f} m/s")
    model.eval()
    with torch.no_grad():
        pred = model(Xte_n).cpu()
    n_params = sum(p.numel() for p in model.parameters())
    return model, pred, Yte, Cte, Ytr, n_params, (mu, sd)


results = {}
for H in (1, 5):
    print(f"Training estimator with history H={H} ({H / 50 * 1000:.0f} ms)")
    model, pred, Yte, Cte, Ytr, n_params, norm = train_eval(H)
    results[f"mlp_H{H}"] = {"rmse_xyz": rmse(pred, Yte), "params": n_params, "test_samples": len(Yte)}
    if H == 5:
        best = (model, pred, Yte, Cte, Ytr, norm)

model, pred, Yte, Cte, Ytr, (mu, sd) = best
results["baseline_mean"] = {"rmse_xyz": rmse(Ytr.mean(0).expand_as(Yte), Yte)}
cmd_pred = torch.cat([Cte[:, :2], torch.zeros(len(Cte), 1)], dim=-1)
results["baseline_command"] = {"rmse_xyz": rmse(cmd_pred, Yte)}

print("\nTest-set RMSE per axis (m/s)   vx      vy      vz")
for k, r in results.items():
    x, y, z = r["rmse_xyz"]
    print(f"  {k:<22} {x:7.4f} {y:7.4f} {z:7.4f}")

with open(os.path.join(args.out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "history": 5}, os.path.join(args.out_dir, "estimator.pt"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    k = torch.randperm(len(Yte))[:4000]
    ax[0].scatter(Yte[k, 0], pred[k, 0], s=2, alpha=0.3)
    lo, hi = float(Yte[:, 0].min()), float(Yte[:, 0].max())
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set_xlabel("True forward velocity vx (m/s)")
    ax[0].set_ylabel("Estimated vx (m/s)")
    ax[0].set_title(f"Estimator on unseen robots (RMSE {results['mlp_H5']['rmse_xyz'][0]:.3f} m/s)")
    names = ["mean", "command", "MLP H=1", "MLP H=5"]
    vals = [results[k_]["rmse_xyz"][0] for k_ in ["baseline_mean", "baseline_command", "mlp_H1", "mlp_H5"]]
    ax[1].bar(names, vals)
    ax[1].set_ylabel("vx RMSE (m/s)")
    ax[1].set_title("Forward-velocity error: baselines vs estimator")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "velocity_estimator.png"), dpi=150)
    print(f"Plot saved to {os.path.join(args.out_dir, 'velocity_estimator.png')}")
except ImportError:
    print("matplotlib not available; skipped plot (run with: uv run --with matplotlib ...)")

print(f"Saved results and model to {args.out_dir}")
