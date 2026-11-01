# Blind Quadruped Locomotion with PPO

Velocity-tracking policies for the Unitree Go2 trained with PPO in **Isaac Lab 3.0** (Newton physics, rsl_rl), on a single **8 GB laptop GPU**. The project asks one question that matters for real robots: **what happens when a locomotion policy cannot measure its own velocity?**

Built as preparation for the Inter IIT Tech Meet 15.0 problem statement on high-speed *blind* locomotion for a wheeled-legged quadruped, where the robot has only joint encoders and an IMU.

## Key finding

The stock Isaac Lab velocity task feeds the policy the robot's **true base linear velocity** (`base_lin_vel`), which no onboard sensor measures directly. I removed it and measured the cost, then tested whether a privileged critic can recover it.

| Variant | Actor sees velocity | Critic sees velocity | Velocity error | Yaw-rate error | Fall rate | Mean reward |
| --- | --- | --- | --- | --- | --- | --- |
| A. Privileged (stock) | Yes | Yes | **0.117 m/s** | 0.158 rad/s | 2.5% | 33.2 |
| B. Blind | No | No | 0.172 m/s | 0.166 rad/s | 2.5% | 30.7 |
| C. Blind + asymmetric critic | No | Yes | **0.139 m/s** | 0.152 rad/s | 2.2% | 32.3 |

All runs: 4,096 parallel environments, 1,500 PPO iterations (~147M environment steps), flat terrain, same seed and hyperparameters. Values are from the final training iteration.

**What this shows**

- **Blindness costs tracking, not stability.** Removing velocity left the fall rate unchanged but raised velocity-tracking error by about 47%.
- **A privileged critic recovers about 60% of that loss with no new sensor.** In variant C the deployed actor sees exactly the same inputs as in B. Only the critic, which exists during training alone, receives the true velocity, giving more accurate advantage estimates.
- **The remaining gap (0.139 vs 0.117 m/s) is what an explicit learned velocity estimator would target**, the next step of this project.

## Training length

| Run | Iterations | Environments | Fall rate | Velocity error | Time |
| --- | --- | --- | --- | --- | --- |
| Short (default) | 300 | 2,048 | 18.3% | 0.54 m/s | 2.9 min |
| Full | 1,500 | 4,096 | 2.5% | 0.12 m/s | 17 min |

The default 300 iterations leaves the policy undertrained; 1,500 iterations cuts falls by 7x and tracking error by 4.5x.

## Setup

| Item | Value |
| --- | --- |
| Simulator | Isaac Lab 3.0, Newton physics (MuJoCo-Warp solver), no Isaac Sim |
| Algorithm | PPO (rsl_rl): clip 0.2, gamma 0.99, GAE lambda 0.95, 5 epochs x 4 minibatches, adaptive KL learning rate |
| Networks | Actor and critic MLP [128, 128, 128], ELU (~41k actor parameters) |
| Control | 12 joint-position targets at 50 Hz, PD control underneath |
| Hardware | RTX 5050 Laptop GPU (8 GB), Windows 11 |

**Actor observations (blind variant, 45 dims):** base angular velocity, projected gravity, velocity command, joint positions, joint velocities, previous action. All are measurable with an IMU and joint encoders.

## Repository contents

```
configs/   git diffs of each variant against stock Isaac Lab 3.0
notes/     ppo_annotated.py (rsl_rl PPO with block-by-block notes)
           PPO_Explained_Go2.pdf (2-page explainer)
media/     gait video and training curves
```

## Reproduce

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git --branch release/3.0.0
cd IsaacLab
git apply path\to\configs\<variant>.diff     # skip for the privileged baseline
uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp --num_envs 4096 --max_iterations 1500
uv run isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 physics=newton_mjwarp --checkpoint latest --viz newton_gl --num_envs 16
```

## Limitations

- One seed per variant; differences of a few percent (falls, yaw) are within run-to-run noise.
- Flat ground with the default, mild domain randomization. The gap is likely larger on rough terrain and under real disturbances.
- Evaluated in simulation only; no hardware deployment.
- The default randomization is much weaker than the target problem statement (10 kg payload, COM shifts up to 14 cm, 250 N pushes, unknown friction).

## Next steps

1. A learned velocity estimator trained concurrently with PPO (after Ji et al. 2022, DreamWaQ), to close the remaining gap.
2. Domain randomization matched to the target disturbances: payload, COM offset, friction, push forces.
3. Rough terrain with the height scan removed, to keep the policy blind.
4. Multiple seeds per variant for confidence intervals.


