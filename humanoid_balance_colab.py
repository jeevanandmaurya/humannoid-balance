"""Colab-friendly Humanoid-v5 balance training.

Run in Colab after installing dependencies and mounting Drive:
    python humanoid_balance_colab.py

All checkpoints, videos, screenshots, and CSV/JSON metrics are written to
Google Drive under MyDrive/humanoid_balance_colab.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


# Matches the observed Colab Free runtime (2 vCPUs, 12 GB RAM).
# MuJoCo is CPU-bound, so using more environments than CPU cores can be slower.
N_ENVS = 2
TOTAL_STEPS = 5_000_000
EPISODE_LENGTH = 1000
N_STEPS = 512
BATCH_SIZE = 256
N_EPOCHS = 5
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 500_000
VIDEO_EPISODES = 5
VIDEO_FPS = 20

PUSH_PHASE2 = 0.10
PUSH_PHASE3 = 0.30
PUSH_LIGHT = 50.0
PUSH_STRONG = 150.0
PUSH_INTERVAL = 200
PUSH_DURATION = 5
BALANCE_GRACE_STEPS = 100

BALANCE_ALIVE = 5.0
BALANCE_UPRIGHT = 3.0
BALANCE_HEIGHT = 2.0
BALANCE_FEET = 1.0
BALANCE_VEL_COST = 0.10
BALANCE_ANG_VEL_COST = 0.05
BALANCE_CTRL_COST = 0.05
BALANCE_SMOOTH_COST = 0.02
TARGET_KL = 0.03

DRIVE_DIR = Path(os.environ.get(
    "HUMANOID_RUN_DIR", "/content/drive/MyDrive/humanoid_balance_colab"))

# CPU is the default for PPO + MlpPolicy. Set HUMANOID_USE_GPU=1 to benchmark
# a GPU runtime such as a T4.
USE_GPU = os.environ.get("HUMANOID_USE_GPU", "0") == "1"
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"


class HumanoidBalanceEnv(gym.Wrapper):
    """Gymnasium Humanoid-v5 with a stationary push-recovery objective."""

    def __init__(self, render_mode=None, push_magnitude=0.0):
        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode,
            forward_reward_weight=0.0,
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            healthy_reward=0.0,
            terminate_when_unhealthy=False,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=EPISODE_LENGTH)
        super().__init__(env)
        self.push_magnitude = push_magnitude
        self._torso_id = self._model.body("torso").id
        self._right_foot_geom = self._model.geom("right_foot").id
        self._left_foot_geom = self._model.geom("left_foot").id
        self._push_force = np.zeros(3)
        self._push_remaining = 0
        self._step_count = 0
        self._previous_action = np.zeros(env.action_space.shape, dtype=np.float32)

        base_dim = env.observation_space.shape[0]
        high = np.full(
            base_dim + 2 + env.action_space.shape[0],
            np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @property
    def _model(self):
        return self.env.unwrapped.model

    @property
    def _data(self):
        return self.env.unwrapped.data

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._push_force[:] = 0.0
        self._push_remaining = 0
        self._step_count = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return self._aug(obs), info

    def trigger_push(self):
        if self.push_magnitude <= 0:
            return
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._push_force = np.array([
            np.cos(angle) * self.push_magnitude,
            np.sin(angle) * self.push_magnitude,
            0.0,
        ])
        self._push_remaining = PUSH_DURATION

    def _push(self):
        if self.push_magnitude > 0 and self._step_count > 0:
            if self._step_count % PUSH_INTERVAL == 0:
                self.trigger_push()
        if self._push_remaining > 0:
            self._data.xfrc_applied[self._torso_id, :3] = self._push_force
            self._push_remaining -= 1
        else:
            self._data.xfrc_applied[self._torso_id, :3] = 0.0

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action_delta = action - self._previous_action
        self._push()
        obs, _, _, truncated, info = self.env.step(action)
        self._step_count += 1
        torso_z = float(self._data.xpos[self._torso_id, 2])
        torso_up_z = float(self._data.xmat[self._torso_id].reshape(3, 3)[2, 2])
        right_foot_z = float(self._data.geom_xpos[self._right_foot_geom, 2])
        left_foot_z = float(self._data.geom_xpos[self._left_foot_geom, 2])
        height_reward = np.exp(-10.0 * (torso_z - 1.4) ** 2)
        feet_reward = np.exp(-40.0 * (
            (right_foot_z - 0.17) ** 2 + (left_foot_z - 0.17) ** 2))
        fallen = torso_z < 0.8 or torso_up_z < math.cos(math.radians(60))
        reward = (
            (BALANCE_ALIVE if not fallen else 0.0)
            + BALANCE_UPRIGHT * torso_up_z
            + BALANCE_HEIGHT * height_reward
            + BALANCE_FEET * feet_reward
            - BALANCE_VEL_COST * float(np.sum(self._data.qvel[:3] ** 2))
            - BALANCE_ANG_VEL_COST * float(np.sum(self._data.qvel[3:6] ** 2))
            - BALANCE_CTRL_COST * float(np.sum(action ** 2))
            - BALANCE_SMOOTH_COST * float(np.sum(np.abs(action_delta)))
        )
        terminated = bool(self._step_count >= BALANCE_GRACE_STEPS and fallen)
        info.update({
            "balance_reward": float(reward),
            "torso_height": torso_z,
            "torso_up_z": torso_up_z,
            "push_magnitude": float(np.linalg.norm(self._data.xfrc_applied[self._torso_id, :3])),
            "action_smoothness": float(np.sum(np.abs(action_delta))),
        })
        self._previous_action = action.copy()
        return self._aug(obs), float(reward), terminated, truncated, info

    def _aug(self, obs):
        return np.concatenate((obs, self._push_force[:2], self._previous_action)).astype(np.float32)


def make_env():
    return HumanoidBalanceEnv()


def make_training_vec():
    raw = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    return raw, VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0)


def curriculum_push(step: int) -> float:
    frac = step / TOTAL_STEPS
    if frac < PUSH_PHASE2:
        return 0.0
    if frac < PUSH_PHASE3:
        return PUSH_LIGHT * (frac - PUSH_PHASE2) / (PUSH_PHASE3 - PUSH_PHASE2)
    return PUSH_LIGHT + (PUSH_STRONG - PUSH_LIGHT) * (
        frac - PUSH_PHASE3) / (1.0 - PUSH_PHASE3)


def resolve_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    files = list(checkpoint_dir.glob("ppo_*_steps.zip"))
    if not files:
        return None
    return max(files, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


def export_checkpoint(model, vec, step: int, root: Path) -> None:
    ckpt_dir = root / "checkpoints"
    out_dir = root / "evaluations" / f"step_{step:09d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / f"ppo_{step}_steps"
    norm_path = ckpt_dir / f"ppo_{step}_steps_vecnormalize.pkl"
    model.save(str(model_path))
    vec.save(str(norm_path))

    eval_env = HumanoidBalanceEnv(
        render_mode="rgb_array", push_magnitude=PUSH_STRONG)
    eval_raw = DummyVecEnv([lambda: eval_env])
    eval_vec = VecNormalize.load(str(norm_path), eval_raw)
    eval_vec.training = False
    eval_vec.norm_reward = False
    eval_model = PPO.load(str(model_path) + ".zip", env=eval_vec, device=DEVICE)
    metrics = []
    try:
        for ep in range(1, VIDEO_EPISODES + 1):
            obs = eval_vec.reset()
            frames = [eval_env.render()]
            total_reward = 0.0
            steps = 0
            while steps < EPISODE_LENGTH:
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_vec.step(action)
                total_reward += float(reward[0])
                steps += 1
                frames.append(eval_env.render())
                if done[0]:
                    break
            prefix = out_dir / f"episode_{ep:02d}"
            iio.imwrite(prefix.with_name(prefix.name + "_initial.png"), frames[0])
            iio.imwrite(prefix.with_name(prefix.name + "_final.png"), frames[-1])
            iio.imwrite(
                prefix.with_suffix(".mp4"), np.stack(frames),
                fps=VIDEO_FPS, codec="libx264",
            )
            metrics.append({"episode": ep, "reward": total_reward, "steps": steps})
            print(f"    episode {ep}/{VIDEO_EPISODES}: reward={total_reward:.1f}, steps={steps}")
    finally:
        eval_vec.close()
        eval_env.close()
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  artifacts saved: {out_dir}")


class CheckpointVideoCallback(BaseCallback):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.next_step = CHECKPOINT_EVERY

    def _on_step(self) -> bool:
        push = curriculum_push(self.num_timesteps)
        self.training_env.env_method("__setattr__", "push_magnitude", push)
        if self.n_calls % 1000 == 0:
            phase = 1 if push == 0 else (2 if push < PUSH_LIGHT else 3)
            print(f"  curriculum: phase={phase}, push={push:.1f} N")
        if self.num_timesteps >= self.next_step:
            step = self.num_timesteps
            print(f"\n[checkpoint] exporting step {step}")
            export_checkpoint(self.model, self.training_env, step, self.root)
            self.next_step += CHECKPOINT_EVERY
        return True


def train(root: Path = DRIVE_DIR):
    root.mkdir(parents=True, exist_ok=True)
    raw, vec = make_training_vec()
    model = PPO(
        "MlpPolicy", vec, device=DEVICE, verbose=1,
        learning_rate=LEARNING_RATE, n_steps=N_STEPS,
        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=1e-3,
        target_kl=TARGET_KL,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256]),
            activation_fn=__import__("torch.nn", fromlist=["ELU"]).ELU,
        ),
        tensorboard_log=str(root / "tensorboard"),
    )
    try:
        if DEVICE == "cuda":
            print(f"Training {N_ENVS} parallel Humanoid-v5 environments with "
                  f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Training {N_ENVS} parallel Humanoid-v5 environments on CPU")
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=CheckpointVideoCallback(root),
            progress_bar=True,
        )
        model.save(str(root / "final_model"))
        vec.save(str(root / "final_vecnormalize.pkl"))
        print(f"Final model saved to {root}")
    finally:
        vec.close()
        raw.close()


if __name__ == "__main__":
    train()
