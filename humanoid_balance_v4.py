# Humanoid Standing Balance & Push Recovery - V4
# =================================================
# Stack : Gymnasium Humanoid-v5 + Stable-Baselines3 PPO
# Device: CPU-friendly configuration; GPU is optional
#
# V4 training features:
#   - Learns natural standing before disturbance training through a standing gate.
#   - Rewards the reset posture and penalizes joint motion, action energy, and drift.
#   - Adds a settled-posture bonus to reduce continuous corrective movement.
#   - Uses a horizontal-push curriculum from 50N to 300N.
#   - Trains push frequency progressively: 1x early, 2x middle, and 3x late.
#   - Automatically resumes from the newest checkpoint in runs/humanoid_balance_v4.
#   - Draws the reset location and active push in the watch viewer.
#
# Run training (automatically resumes when a checkpoint exists):
#     python humanoid_balance_v4.py
#
# Watch the newest saved checkpoint:
#     python humanoid_balance_v4.py --watch
#
# Evaluate the newest saved checkpoint once:
#     python humanoid_balance_v4.py --eval

# Install:
#     pip install gymnasium[mujoco] stable-baselines3 tensorboard torch


import argparse
import math
import re
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# v4 adds natural-posture and low-motion shaping, so its checkpoints are not compatible
# with the older 350-value-observation checkpoints.
LOG_DIR        = Path("./runs/humanoid_balance_v4")
N_ENVS         = 12
TOTAL_STEPS    = 50_000_000
EPISODE_LENGTH = 1000

# PPO
LR          = 1e-4
N_STEPS     = 1024
BATCH_SIZE  = 512
N_EPOCHS    = 5
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_RANGE  = 0.2
ENT_COEF    = 1e-3

# Push curriculum
PUSH_PHASE2   = 0.10   # fraction of TOTAL_STEPS where light push starts
PUSH_PHASE3   = 0.30   # fraction where strong push starts
PUSH_LIGHT    = 50.0   # N
PUSH_STRONG   = 300.0  # N
PUSH_INTERVAL = 200    # steps between impulses
PUSH_DURATION = 5      # steps impulse lasts

# Disturbance frequency curriculum: begin with the familiar 1x frequency,
# then progressively train recovery from 2x and 3x more frequent pushes.
PUSH_FREQ_EARLY = 1.0
PUSH_FREQ_MIDDLE = 2.0
PUSH_FREQ_LATE = 3.0

# Balance objective (the default Humanoid-v5 reward is locomotion-oriented).
BALANCE_ALIVE = 5.0
BALANCE_UPRIGHT = 3.0
BALANCE_HEIGHT = 2.0
BALANCE_FEET = 1.0
BALANCE_VEL_COST = 0.10
BALANCE_ANG_VEL_COST = 0.05
BALANCE_SMOOTH_COST = 0.02  # temporal action-change penalty
BALANCE_DRIFT_COST = 5.0    # keep the torso near its reset position
BALANCE_POSTURE = 2.0       # return joints toward the natural reset posture
BALANCE_JOINT_VEL_COST = 0.03
BALANCE_ACTION_COST = 0.05
BALANCE_SETTLED_BONUS = 1.0
TARGET_KL = 0.03
STANDING_GATE_LENGTH = 800  # do not add pushes until standing is reliable
BALANCE_CTRL_COST = 0.05
BALANCE_GRACE_STEPS = 100  # allow recovery before early fall termination


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HumanoidBalanceEnv(gym.Wrapper):
    """
    Wraps Gymnasium Humanoid-v5.
    Adds random horizontal push disturbances + push_force_xy appended to obs.
    All reward, termination, and base obs come from Humanoid-v5 unchanged.
    """

    def __init__(self, render_mode=None, push_magnitude=0.0):
        # Disable the built-in locomotion reward/termination.  The wrapper
        # supplies an explicit stationary balance objective below.
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

        base_dim = env.observation_space.shape[0]
        high = np.full(
            base_dim + 4 + env.action_space.shape[0],
            np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._push_force     = np.zeros(3)
        self._push_remaining = 0
        self._step_count     = 0
        self._previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
        self._target_xy = np.zeros(2, dtype=np.float32)
        self._target_qpos = np.zeros(self._data.qpos.shape, dtype=np.float64)
        self.push_interval = PUSH_INTERVAL

    @property
    def _model(self): return self.env.unwrapped.model

    @property
    def _data(self): return self.env.unwrapped.data

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._push_force     = np.zeros(3)
        self._push_remaining = 0
        self._step_count     = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._target_xy = self._data.xpos[self._torso_id, :2].copy()
        self._target_qpos = self._data.qpos.copy()
        return self._aug(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action_delta = action - self._previous_action
        self._push()
        obs, _, _, truncated, info = self.env.step(action)
        self._step_count += 1

        data = self._data
        torso_z = float(data.xpos[self._torso_id, 2])
        torso_up_z = float(data.xmat[self._torso_id].reshape(3, 3)[2, 2])
        right_foot_z = float(data.geom_xpos[self._right_foot_geom, 2])
        left_foot_z = float(data.geom_xpos[self._left_foot_geom, 2])

        # Stationary balance reward: survive, stay upright and at nominal
        # height, keep both feet near the floor, and avoid violent motion.
        height_reward = np.exp(-10.0 * (torso_z - 1.4) ** 2)
        feet_reward = np.exp(-40.0 * (
            (right_foot_z - 0.17) ** 2 + (left_foot_z - 0.17) ** 2))
        drift_xy = data.xpos[self._torso_id, :2] - self._target_xy
        drift_distance = float(np.linalg.norm(drift_xy))
        posture_error = float(np.mean(np.square(data.qpos[7:] - self._target_qpos[7:])))
        joint_velocity = float(np.mean(np.square(data.qvel[6:])))
        posture_reward = np.exp(-3.0 * posture_error)
        fallen = torso_z < 0.8 or torso_up_z < np.cos(np.deg2rad(60.0))
        reward = (
            (BALANCE_ALIVE if not fallen else 0.0)
            + BALANCE_UPRIGHT * torso_up_z
            + BALANCE_HEIGHT * height_reward
            + BALANCE_FEET * feet_reward
            - BALANCE_VEL_COST * float(np.sum(np.square(data.qvel[:3])))
            - BALANCE_ANG_VEL_COST * float(np.sum(np.square(data.qvel[3:6])))
            - BALANCE_CTRL_COST * float(np.sum(np.square(action)))
            - BALANCE_ACTION_COST * float(np.mean(np.square(action)))
            - BALANCE_SMOOTH_COST * float(np.sum(np.abs(action_delta)))
            - BALANCE_DRIFT_COST * float(np.sum(np.square(drift_xy)))
            + BALANCE_POSTURE * posture_reward
            - BALANCE_JOINT_VEL_COST * joint_velocity
        )

        settled = bool(
            not fallen
            and posture_error < 0.03
            and np.linalg.norm(data.qvel[6:]) < 1.0
            and float(np.mean(np.abs(action_delta))) < 0.08
        )
        if settled:
            reward += BALANCE_SETTLED_BONUS

        terminated = bool(
            self._step_count >= BALANCE_GRACE_STEPS and fallen
        )
        info.update({
            "balance_reward": float(reward),
            "torso_height": torso_z,
            "torso_up_z": torso_up_z,
            "feet_reward": float(feet_reward),
            "action_smoothness": float(np.sum(np.abs(action_delta))),
            "drift_distance": drift_distance,
            "posture_error": posture_error,
            "joint_velocity": joint_velocity,
            "settled": settled,
            # DummyVecEnv resets immediately after done, so preserve the
            # terminal state for watch-mode reporting.
            "episode_sim_time": float(data.time),
            "episode_fallen": bool(fallen),
        })
        self._previous_action = action.copy()
        return self._aug(obs), float(reward), terminated, truncated, info

    def _push(self):
        if self.push_magnitude <= 0:
            return
        # Give the policy time to establish its stance before the first
        # disturbance; applying 150 N at step 0 makes early checkpoints look
        # like instant failures instead of showing baseline balance.
        if self._step_count > 0 and self._step_count % self.push_interval == 0:
            self.trigger_push()
        torso_id = self._model.body("torso").id
        if self._push_remaining > 0:
            self._data.xfrc_applied[torso_id, :3] = self._push_force
            self._push_remaining -= 1
        else:
            self._data.xfrc_applied[torso_id, :3] = 0.0

    def trigger_push(self):
        """Apply a new random horizontal push immediately."""
        if self.push_magnitude <= 0:
            return
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._push_force = np.array([
            np.cos(angle) * self.push_magnitude,
            np.sin(angle) * self.push_magnitude,
            0.0,
        ])
        self._push_remaining = PUSH_DURATION
        torso_id = self._model.body("torso").id
        self._data.xfrc_applied[torso_id, :3] = self._push_force

    def _aug(self, obs):
        drift_xy = self._data.xpos[self._torso_id, :2] - self._target_xy
        return np.concatenate((obs, self._push_force[:2], drift_xy,
                               self._previous_action)).astype(np.float32)


def _draw_push(viewer, env: HumanoidBalanceEnv) -> None:
    """Draw the spawn circle and currently applied torso push arrow."""
    scene = viewer.user_scn
    scene.ngeom = 0
    center = env._target_xy
    radius = 0.75
    for i in range(32):
        if scene.ngeom >= scene.maxgeom:
            break
        angle = 2.0 * np.pi * i / 32.0
        pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            0.02,
        ], dtype=float)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0.0, 0.0]), pos, np.eye(3).reshape(-1),
            np.array([0.1, 1.0, 0.2, 0.9], dtype=np.float32),
        )
        scene.ngeom += 1
    force = np.asarray(env._data.xfrc_applied[env._torso_id, :3], dtype=float)
    magnitude = float(np.linalg.norm(force))
    if magnitude < 1e-6 or scene.maxgeom < 1:
        return

    direction = force / magnitude
    position = env._data.xpos[env._torso_id].copy() + direction * 0.12
    length = 0.25 + min(magnitude / 500.0, 0.5)
    size = np.array([0.035, 0.035, length], dtype=float)
    quat = np.zeros(4, dtype=float)
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quatZ2Vec(quat, direction)
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
        size, position, mat, np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_watch_status(viewer, env: HumanoidBalanceEnv, text: str) -> None:
    """Draw watch statistics in a fixed top-left screen overlay."""
    viewer.set_texts((
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        text,
        "",
    ))


def _hide_native_force_visuals(viewer) -> None:
    """Keep only the custom red push arrow visible."""
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def curriculum_push(step: int) -> float:
    """Return the push magnitude for a training/evaluation timestep."""
    frac = min(max(step / TOTAL_STEPS, 0.0), 1.0)
    if frac < PUSH_PHASE2:
        return 0.0
    if frac < PUSH_PHASE3:
        return PUSH_LIGHT * (frac - PUSH_PHASE2) / (PUSH_PHASE3 - PUSH_PHASE2)
    return PUSH_LIGHT + (PUSH_STRONG - PUSH_LIGHT) * (
        (frac - PUSH_PHASE3) / (1.0 - PUSH_PHASE3))


def curriculum_push_interval(step: int) -> int:
    """Use 1x, then 2x, then 3x the base push frequency."""
    frac = min(max(step / TOTAL_STEPS, 0.0), 1.0)
    if frac < PUSH_PHASE2:
        multiplier = PUSH_FREQ_EARLY
    elif frac < PUSH_PHASE3:
        multiplier = PUSH_FREQ_MIDDLE
    else:
        multiplier = PUSH_FREQ_LATE
    return max(1, int(round(PUSH_INTERVAL / multiplier)))


class CurriculumCallback(BaseCallback):
    """Ramps the same push curriculum in training and evaluation."""

    def __init__(self, raw_vec, eval_vec, eval_callback):
        super().__init__()
        self._raw = raw_vec
        self._eval = eval_vec
        self._eval_callback = eval_callback

    def _on_step(self):
        frac = min(self.num_timesteps / TOTAL_STEPS, 1.0)
        scheduled_mag = curriculum_push(self.num_timesteps)
        scheduled_interval = curriculum_push_interval(self.num_timesteps)
        standing_ready = False
        if self._eval_callback.evaluations_length:
            standing_ready = (
                float(np.mean(self._eval_callback.evaluations_length[-1]))
                >= STANDING_GATE_LENGTH
            )
        # Never introduce disturbances before the no-push standing skill has
        # been demonstrated by evaluation.
        mag = scheduled_mag if standing_ready else 0.0
        self._raw.env_method("__setattr__", "push_magnitude", mag)
        self._raw.env_method("__setattr__", "push_interval", scheduled_interval)
        self._eval.env_method("__setattr__", "push_magnitude", mag)
        self._eval.env_method("__setattr__", "push_interval", scheduled_interval)
        if self.n_calls % 10_000 == 0:
            phase = 1 if frac < PUSH_PHASE2 else (2 if frac < PUSH_PHASE3 else 3)
            print(f"  [{frac*100:4.1f}%] phase={phase}  push={mag:.0f} N  "
                  f"steps={self.num_timesteps:,}  interval={scheduled_interval}  "
                  f"standing_gate={'READY' if standing_ready else 'WAIT'}")
        return True


class BestVecNormalizeCallback(BaseCallback):
    """Save normalization statistics whenever EvalCallback finds a new best."""

    def __init__(self, eval_callback):
        super().__init__()
        self.eval_callback = eval_callback
        self._last_best = -np.inf

    def _on_step(self):
        best = self.eval_callback.best_mean_reward
        if best > self._last_best:
            best_dir = Path(self.eval_callback.best_model_save_path)
            best_dir.mkdir(parents=True, exist_ok=True)
            self.training_env.save(str(best_dir / "best_model_vecnormalize.pkl"))
            self._last_best = best
        return True


class LiveTestCallback(BaseCallback):
    """
    Every `test_every` training steps, opens a viewer window and runs
    the current policy for `n_steps` steps so you can watch it live.
    Closes automatically and training resumes.
    """

    def __init__(self, test_every=100_000, n_steps=300):
        super().__init__()
        self.test_every = test_every
        self.n_steps    = n_steps
        self._last      = 0

    def _on_step(self):
        if self.num_timesteps - self._last < self.test_every:
            return True
        self._last = self.num_timesteps
        print(f"\n  [viewer] opening for {self.n_steps} steps ...")
        env = HumanoidBalanceEnv(render_mode=None,
                                 push_magnitude=curriculum_push(self.num_timesteps))
        obs, _ = env.reset()
        total_r = 0.0
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            for _ in range(self.n_steps):
                if not viewer.is_running():
                    break
                step_start = time.perf_counter()
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action)
                total_r += r
                _draw_push(viewer, env)
                viewer.sync()
                _hide_native_force_visuals(viewer)
                remaining = env.unwrapped.dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
                if terminated or truncated:
                    # Early policies fall quickly.  Reset the same model and
                    # keep the same viewer open for the complete test.
                    obs, _ = env.reset()
        env.close()
        print(f"  [viewer] reward = {total_r:.1f}\n")
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_training_checkpoint():
    """Return the newest numbered PPO checkpoint, if one exists."""
    checkpoint_dir = LOG_DIR / "ckpts"
    saved = list(checkpoint_dir.glob("ppo_*_steps.zip"))
    if not saved:
        return None

    def step_number(path):
        match = re.search(r"ppo_(\d+)_steps\.zip$", path.name)
        return int(match.group(1)) if match else -1

    return max(saved, key=step_number)


def checkpoint_vecnormalize_path(model_path):
    """Find the VecNormalize file saved alongside a PPO checkpoint."""
    match = re.match(r"ppo_(\d+)_steps\.zip$", model_path.name)
    if match:
        candidate = model_path.with_name(
            f"ppo_vecnormalize_{match.group(1)}_steps.pkl")
        if candidate.exists():
            return candidate

    candidate = model_path.with_name(
        model_path.stem + "_vecnormalize.pkl")
    return candidate if candidate.exists() else None


def make_raw_vec(n_envs):
    return SubprocVecEnv([
        (lambda: HumanoidBalanceEnv()) for _ in range(n_envs)
    ])

def make_vec(n_envs, training=True):
    raw = make_raw_vec(n_envs)
    vec = VecNormalize(raw, norm_obs=True,
                       norm_reward=training, clip_obs=10.0)
    return raw, vec


def make_eval_vec(norm_ref=None, push_magnitude=0.0):
    raw = DummyVecEnv([lambda: Monitor(HumanoidBalanceEnv(
        push_magnitude=push_magnitude))])
    vec = VecNormalize(raw, norm_obs=True, norm_reward=False,
                       clip_obs=10.0, training=False)
    if norm_ref is not None:
        vec.obs_rms = norm_ref.obs_rms
    return vec


def build_model(env):
    import torch.nn as nn
    return PPO(
        "MlpPolicy", env,
        learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, ent_coef=ENT_COEF, target_kl=TARGET_KL,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256, 256],
                          vf=[256, 256, 256, 256]),
            activation_fn=nn.ELU,
        ),
        tensorboard_log=str(LOG_DIR / "tb"),
        device="cpu",
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(live_render=False, resume=True):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training  envs={N_ENVS}  steps={TOTAL_STEPS:,}")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR / 'tb'}\n")

    raw, vec  = make_vec(N_ENVS)
    eval_vec  = make_eval_vec(norm_ref=vec, push_magnitude=0.0)

    checkpoint = latest_training_checkpoint() if resume else None
    if checkpoint is not None:
        norm_path = checkpoint_vecnormalize_path(checkpoint)
        if norm_path is None:
            raise FileNotFoundError(
                f"VecNormalize statistics missing for checkpoint: {checkpoint}")

        vec.close()
        raw = make_raw_vec(N_ENVS)
        vec = VecNormalize.load(str(norm_path), raw)
        vec.training = True
        vec.norm_reward = True
        eval_vec.close()
        eval_vec = make_eval_vec(norm_ref=vec, push_magnitude=0.0)

        model = PPO.load(str(checkpoint), env=vec, device="cpu")
        print(f"Resuming from: {checkpoint}")
        print(f"Restored normalization: {norm_path}")
        print(f"Previously trained steps: {model.num_timesteps:,}\n")
    else:
        model = build_model(vec)
        print("No checkpoint found; starting a new training run.\n")

    eval_callback = EvalCallback(
            eval_vec,
            best_model_save_path=str(LOG_DIR / "best"),
            log_path=str(LOG_DIR / "eval"),
            eval_freq=max(500_000 // N_ENVS, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        )
    callbacks = [
        CurriculumCallback(raw, eval_vec, eval_callback),
        eval_callback,
        BestVecNormalizeCallback(eval_callback),
        CheckpointCallback(
            save_freq=max(1_000_000 // N_ENVS, 1),
            save_path=str(LOG_DIR / "ckpts"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
    ]
    if live_render:
        callbacks.append(LiveTestCallback(test_every=100_000, n_steps=300))

    remaining_steps = TOTAL_STEPS - model.num_timesteps
    print(f"Training progress: {min(model.num_timesteps, TOTAL_STEPS):,} / "
          f"{TOTAL_STEPS:,} steps")
    print(f"Remaining: {max(remaining_steps, 0):,} steps\n")
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=callbacks,
                    reset_num_timesteps=False, progress_bar=True)
    else:
        print("Training target already reached; saving the loaded model.")

    model.save(str(LOG_DIR / "final_model"))
    vec.save(str(LOG_DIR / "vecnormalize.pkl"))
    print(f"\nSaved to {LOG_DIR}")
    vec.close()
    eval_vec.close()


# ---------------------------------------------------------------------------
# Eval / render
# ---------------------------------------------------------------------------

def watch(checkpoint=None, n_episodes=3, continuous=False):
    """Watch a saved model independently of the training process."""
    if checkpoint:
        requested = Path(checkpoint)
    else:
        checkpoint_dir = LOG_DIR / "ckpts"
        saved = list(checkpoint_dir.glob("ppo_*_steps.zip"))
        if saved:
            def step_number(path):
                match = re.search(r"ppo_(\d+)_steps\.zip$", path.name)
                return int(match.group(1)) if match else -1
            requested = max(saved, key=step_number)
        elif (LOG_DIR / "best" / "best_model.zip").exists():
            requested = LOG_DIR / "best"
        else:
            requested = LOG_DIR / "final_model.zip"
    if requested.is_dir():
        if (requested / "best_model.zip").exists():
            model_path = requested / "best_model.zip"
        elif (requested / "final_model.zip").exists():
            model_path = requested / "final_model.zip"
        else:
            raise FileNotFoundError(f"No model found in {requested}")
    else:
        model_path = requested
        if model_path.suffix != ".zip":
            model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Match the disturbance level used during training for numbered
    # checkpoints.  A directory/final model without a step number uses the
    # strong-push setting as an explicit stress test.
    checkpoint_match = re.match(r"ppo_(\d+)_steps\.zip$", model_path.name)
    watch_push_magnitude = (
        curriculum_push(int(checkpoint_match.group(1)))
        if checkpoint_match else PUSH_STRONG
    )

    # CheckpointCallback saves matching VecNormalize statistics beside each
    # checkpoint.  Fall back to the final stats file when available.
    norm_candidates = [
        model_path.with_name(model_path.stem + "_vecnormalize.pkl"),
        LOG_DIR / "vecnormalize.pkl",
    ]
    # Stable-Baselines3 CheckpointCallback names this file
    # ppo_vecnormalize_<steps>_steps.pkl.
    match = re.match(r"ppo_(\d+)_steps$", model_path.stem)
    if match:
        norm_candidates.insert(
            1, model_path.with_name(
                f"ppo_vecnormalize_{match.group(1)}_steps.pkl"))
    norm_path = next((p for p in norm_candidates if p.exists()), None)

    watched_env = HumanoidBalanceEnv(
        render_mode=None, push_magnitude=watch_push_magnitude)
    raw = DummyVecEnv([lambda: watched_env])
    vec = VecNormalize.load(str(norm_path), raw) if norm_path else raw
    if hasattr(vec, "training"):
        vec.training    = False
        vec.norm_reward = False

    model = PPO.load(str(model_path), env=vec, device="cpu")
    print(f"Watching: {model_path}")
    print(f"Watch push magnitude: {watch_push_magnitude:.1f} N "
          f"(matches checkpoint curriculum)")
    episode = 0
    controls = {"paused": False, "reset": False, "push": True, "ai": True}
    episode_start_sim_time = 0.0
    last_status_wall_time = 0.0

    def key_callback(key: int) -> None:
        nonlocal watch_push_magnitude
        if key == ord(" "):
            controls["paused"] = not controls["paused"]
            print("  [watch]", "paused" if controls["paused"] else "playing")
        elif key in (ord("r"), ord("R")):
            controls["reset"] = True
        elif key in (ord("p"), ord("P")):
            controls["push"] = not controls["push"]
            watched_env.push_magnitude = (
                watch_push_magnitude if controls["push"] else 0.0)
            if not controls["push"]:
                watched_env._push_remaining = 0
            watched_env._data.xfrc_applied[watched_env._model.body("torso").id, :3] = 0.0
            print("  [watch] pushes", "enabled" if controls["push"] else "disabled")
        elif key in (ord("a"), ord("A")):
            controls["ai"] = not controls["ai"]
            print("  [watch] AI", "enabled" if controls["ai"] else "disabled")
        elif key in (ord("f"), ord("F")):
            if controls["push"]:
                watched_env.trigger_push()
                print(f"  [watch] manual push applied: "
                      f"{watch_push_magnitude:.1f} N")
        elif key in (ord("-"), ord("_")):
            watch_push_magnitude = max(0.0, watch_push_magnitude - 10.0)
            if controls["push"]:
                watched_env.push_magnitude = watch_push_magnitude
            print(f"  [watch] push force: {watch_push_magnitude:.1f} N")
        elif key in (ord("="), ord("+")):
            watch_push_magnitude += 10.0
            if controls["push"]:
                watched_env.push_magnitude = watch_push_magnitude
            print(f"  [watch] push force: {watch_push_magnitude:.1f} N")
        elif key in (ord("0"),):
            watch_push_magnitude = 0.0
            watched_env.push_magnitude = 0.0
            watched_env._push_remaining = 0
            print("  [watch] push force: 0.0 N")

    try:
        with mujoco.viewer.launch_passive(
                watched_env._model, watched_env._data,
                key_callback=key_callback) as viewer:
            mujoco.mjv_defaultFreeCamera(watched_env._model, viewer.cam)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = watched_env._data.xpos[watched_env._model.body("torso").id]
            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            while (continuous or episode < n_episodes) and viewer.is_running():
                if controls["paused"]:
                    viewer.sync()
                    time.sleep(0.05)
                    continue
                obs, total_r, done = vec.reset(), 0.0, False
                terminal_info = {}
                episode_start_sim_time = float(watched_env.unwrapped.data.time)
                last_status_wall_time = 0.0
                print(f"\n[watch] Episode {episode + 1} started | "
                      f"AI={'ON' if controls['ai'] else 'OFF'}")
                while not done and viewer.is_running():
                    if controls["reset"]:
                        controls["reset"] = False
                        obs, total_r, done = vec.reset(), 0.0, False
                        terminal_info = {}
                        episode_start_sim_time = float(watched_env.unwrapped.data.time)
                        last_status_wall_time = 0.0
                        print(f"\n[watch] Episode reset | "
                              f"AI={'ON' if controls['ai'] else 'OFF'}")
                    if controls["paused"]:
                        viewer.sync()
                        time.sleep(0.05)
                        continue
                    step_start = time.perf_counter()
                    if controls["ai"]:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        # DummyVecEnv expects a batch dimension: (1, 17).
                        action = np.zeros(
                            (1,) + watched_env.action_space.shape,
                            dtype=np.float32,
                        )
                    obs, r, done, infos = vec.step(action)
                    if bool(done[0]):
                        terminal_info = infos[0]
                    total_r += float(r[0])
                    _draw_push(viewer, watched_env)
                    sim_elapsed = (float(watched_env.unwrapped.data.time)
                                   - episode_start_sim_time)
                    torso_id = watched_env._model.body("torso").id
                    height = float(watched_env._data.xpos[torso_id, 2])
                    torso_up_z = float(watched_env._data.xmat[torso_id].reshape(3, 3)[2, 2])
                    drift = float(np.linalg.norm(
                        watched_env._data.xpos[torso_id, :2]
                        - watched_env._target_xy))
                    applied_force = float(np.linalg.norm(
                        watched_env._data.xfrc_applied[torso_id, :3]))
                    _draw_watch_status(
                        viewer, watched_env,
                        f"AI {'ON' if controls['ai'] else 'OFF'} | "
                        f"TIME {sim_elapsed:.2f}s | "
                        f"HEIGHT {height:.2f}m | "
                        f"UPRIGHT {torso_up_z:.2f} | DRIFT {drift:.2f}m | "
                        f"PUSH {watch_push_magnitude:.0f}N "
                        f"(applied {applied_force:.0f}N)",
                    )
                    viewer.sync()
                    _hide_native_force_visuals(viewer)
                    now = time.perf_counter()
                    if now - last_status_wall_time >= 1.0:
                        print(
                            f"\r[watch] AI={'ON ' if controls['ai'] else 'OFF'} "
                            f"sim_time={sim_elapsed:6.2f}s "
                            f"height={height:5.2f}m "
                            f"push={'ON' if controls['push'] else 'OFF'}",
                            end="", flush=True)
                        last_status_wall_time = now
                    remaining = watched_env.unwrapped.dt - (
                        time.perf_counter() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
                if not viewer.is_running():
                    break
                episode += 1
                if terminal_info:
                    sim_elapsed = float(terminal_info["episode_sim_time"])
                    fallen = bool(terminal_info["episode_fallen"])
                else:
                    torso_id = watched_env._model.body("torso").id
                    torso_z = float(watched_env._data.xpos[torso_id, 2])
                    torso_up_z = float(watched_env._data.xmat[torso_id].reshape(3, 3)[2, 2])
                    fallen = torso_z < 0.8 or torso_up_z < math.cos(math.radians(60))
                    sim_elapsed = float(watched_env.unwrapped.data.time) - episode_start_sim_time
                result = "FELL" if fallen else "TIME LIMIT/STOPPED"
                print(f"\nEpisode {episode}: {result} after "
                      f"{sim_elapsed:.2f} simulated seconds | "
                      f"AI={'ON' if controls['ai'] else 'OFF'} | "
                      f"reward={total_r:.1f}")
    finally:
        vec.close()


def evaluate(checkpoint=None, n_episodes=3):
    """Backward-compatible alias for watch mode."""
    return watch(checkpoint, n_episodes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-test", action="store_true",
                    help="Open viewer every 100k steps during training")
    ap.add_argument("--eval",        action="store_true",
                    help="Watch saved model (skip training)")
    ap.add_argument("--watch",       action="store_true",
                    help="Watch latest saved checkpoint independently")
    ap.add_argument("--checkpoint",  type=str, default=None)
    args = ap.parse_args()

    if args.watch:
        print("Watch mode: press Ctrl+C to stop.")
        try:
            # Keep one viewer and one loaded policy alive.  The humanoid
            # state resets between episodes, but the window is never recreated.
            watch(args.checkpoint, continuous=True)
        except KeyboardInterrupt:
            print("\nWatch stopped.")
    elif args.eval:
        watch(args.checkpoint)
    else:
        train(live_render=args.render_test)
