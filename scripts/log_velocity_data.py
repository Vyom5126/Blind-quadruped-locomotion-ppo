"""Log rollouts from a trained policy for offline velocity-estimator training.

Records, at every policy step:
  - the actor's blind observations (obs["policy"], 45 dims)
  - the true base linear velocity (first 3 dims of obs["critic"], noise-free)
  - episode-done flags (so history windows never cross an episode boundary)

Requires the asymmetric-critic config to be applied (critic group with base_lin_vel first).

Usage (from C:\\rl\\IsaacLab):
  uv run isaaclab -p scripts/log_velocity_data.py --task Isaac-Velocity-Flat-UnitreeGo2 ^
      physics=newton_mjwarp --num_envs 256 --checkpoint <path to model_1499.pt> --steps 1500
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch

# --- our own arguments, stripped before Isaac Lab parses the rest -------------
_own = argparse.ArgumentParser(add_help=False)
_own.add_argument("--steps", type=int, default=1500, help="Policy steps to record (50 Hz).")
_own.add_argument("--out", type=str, default="logs/velocity_data.pt", help="Output file.")
own_args, isaaclab_argv = _own.parse_known_args(sys.argv[1:])

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import DirectMARLEnvCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)


def main() -> None:
    args_cli = _parse_args(isaaclab_argv)
    installed_version = check_rsl_rl_version()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            raise RuntimeError("No checkpoint found. Pass --checkpoint <path to model_XXXX.pt>.")

        env = create_isaaclab_env(
            args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg)
        )
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[LOG] Loading checkpoint: {resume_path}")
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        if "critic" not in obs.keys():
            raise RuntimeError(
                "No 'critic' observation group. Re-apply the asym config diff first "
                "(the critic group carries the true base velocity)."
            )
        print(f"[LOG] policy obs dim: {obs['policy'].shape[-1]}, critic obs dim: {obs['critic'].shape[-1]}")

        pol_buf, vel_buf, done_buf = [], [], []
        with torch.inference_mode():
            for t in range(own_args.steps):
                pol_buf.append(obs["policy"].detach().cpu())
                vel_buf.append(obs["critic"][:, :3].detach().cpu())  # true base_lin_vel (body frame)
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                done_buf.append(dones.detach().cpu().bool())
                if (t + 1) % 250 == 0:
                    print(f"[LOG] {t + 1}/{own_args.steps} steps")

        data = {
            "policy_obs": torch.stack(pol_buf),  # [T, N, 45]
            "true_vel": torch.stack(vel_buf),  # [T, N, 3]
            "dones": torch.stack(done_buf),  # [T, N]  done AFTER step t
            "checkpoint": resume_path,
            "step_dt": float(env.unwrapped.step_dt),
        }
        os.makedirs(os.path.dirname(os.path.abspath(own_args.out)), exist_ok=True)
        torch.save(data, own_args.out)
        print(f"[LOG] Saved {data['policy_obs'].shape} to {os.path.abspath(own_args.out)}")


if __name__ == "__main__":
    main()
