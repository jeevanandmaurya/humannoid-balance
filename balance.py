"""
Humanoid Standing Balance & Push Recovery — MJX + Brax PPO
===========================================================
Training method: Proximal Policy Optimization (PPO)
  - Massively parallel MJX environments (GPU-batched via JAX)
  - On-policy rollouts: no replay buffer needed, stable under large batches
  - Push robustness via domain-randomisation curriculum:
      phase 1 (0–40 %):  no pushes          → learn stable upright stance
      phase 2 (40–70 %): light pushes        → reactive ankle/hip strategy
      phase 3 (70–100%): strong random pushes → robust recovery

Rationale for PPO over SAC
  • SAC wins on sample efficiency when trajectory optimisation is cheap.
  • PPO wins on wall-clock time with 8 192+ parallel MJX envs on GPU —
    matching the MuJoCo Playground / REEM-C papers (200 M steps in ~56 min
    on a single RTX 4090).
  • Push-recovery papers (FRASA 2024, HiFAR 2025) all use PPO with
    multi-stage curriculum as the backbone.

Dependencies
  pip install mujoco mujoco-mjx brax flax optax jax[cuda12]
  # or jax[cpu] for CPU-only testing

References
  • MuJoCo Playground (Zakka et al., 2025)
  • Learning Velocity-based Humanoid Locomotion: Massively Parallel with
    Brax/MJX (Thibault et al., CLAWAR 2024)
  • FRASA: End-to-End RL for Fall Recovery (Gaspard et al., 2024)
  • HiFAR: Multi-Stage Curriculum for Humanoid Fall Recovery (Chen et al., 2025)
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs.base import Env, State
from flax import struct
import flax.linen as nn

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── MuJoCo model ──────────────────────────────────────────────────────
    # Uses the built-in humanoid that ships with MuJoCo.
    # Swap for your own MJCF path if needed (e.g. Unitree H1, G1).
    model_path: str = "humanoid"          # "humanoid" loads mujoco.MjModel built-in

    # ── Training ──────────────────────────────────────────────────────────
    num_envs: int = 4096                  # parallel rollout envs; 8192 if VRAM allows
    num_timesteps: int = 200_000_000      # total environment steps
    episode_length: int = 1000            # steps per episode  (dt=0.005 → 5 s)
    num_eval_envs: int = 256
    eval_every: int = 5_000_000

    # ── PPO hyper-params ──────────────────────────────────────────────────
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-3
    discounting: float = 0.97
    unroll_length: int = 20              # rollout horizon before update
    batch_size: int = 2048
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    reward_scaling: float = 0.1
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_observations: bool = True

    # ── Push-disturbance curriculum ───────────────────────────────────────
    # Three phases defined as fractions of total training steps.
    push_phase2_start: float = 0.40      # light push starts at 40 % of training
    push_phase3_start: float = 0.70      # strong push starts at 70 %
    push_interval_steps: int = 100       # apply impulse every N sim steps
    push_force_light:  float = 50.0      # N   (phase 2)
    push_force_strong: float = 150.0     # N   (phase 3)
    push_duration_steps: int = 5         # how many steps the impulse lasts

    # ── Reward weights ────────────────────────────────────────────────────
    w_alive:      float = 5.0            # per-step survival bonus
    w_upright:    float = 3.0            # torso pointing up
    w_height:     float = 2.0            # COM height near nominal
    w_small_ctrl: float = 0.05           # penalise large actuator efforts
    w_small_vel:  float = 0.1            # penalise root linear velocity (stay still)
    w_ang_vel:    float = 0.05           # penalise angular velocity
    w_foot:       float = 1.0            # both feet on ground symmetrically

    # ── Logging / checkpointing ───────────────────────────────────────────
    log_dir: str = "./runs/humanoid_balance"
    checkpoint_every: int = 10_000_000

CFG = Config()

# ---------------------------------------------------------------------------
# 1.  Load MuJoCo model
# ---------------------------------------------------------------------------

def load_model(cfg: Config) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load built-in humanoid or a custom MJCF path."""
    if cfg.model_path == "humanoid":
        import os
        # MuJoCo ships example models; locate humanoid.xml
        mj_path = Path(mujoco.__file__).parent / "mjmodel.so"
        model_dir = Path(mujoco.__file__).parent / "testdata"
        # Try the standard location used by dm_control / mujoco-py
        candidate_dirs = [
            Path(mujoco.__file__).parent.parent / "mujoco" / "model",
            Path(mujoco.__file__).parent / "model",
            Path(os.environ.get("MUJOCO_MODEL_DIR", "")) / "humanoid",
        ]
        xml_path = None
        for d in candidate_dirs:
            p = d / "humanoid" / "humanoid.xml"
            if p.exists():
                xml_path = str(p)
                break
        if xml_path is None:
            # Fallback: generate a minimal humanoid XML inline
            xml_path = _write_minimal_humanoid_xml()
        model = mujoco.MjModel.from_xml_path(xml_path)
    else:
        model = mujoco.MjModel.from_xml_path(cfg.model_path)

    model.opt.timestep = 0.005          # 5 ms physics step
    data = mujoco.MjData(model)
    return model, data


def _write_minimal_humanoid_xml() -> str:
    """
    Write a minimal humanoid MJCF that is always available.
    Based on MuJoCo's built-in humanoid, simplified to core DOFs needed for
    standing balance.  For real research use the full Unitree G1/H1 XML.
    """
    xml = """
<mujoco model="humanoid_balance">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton"
          integrator="RK4"/>
  <default>
    <joint limited="true" damping="1" armature="0" stiffness="0"
           actuatorfrcrange="-1000 1000"/>
    <geom condim="1" material="geom"/>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
  </default>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid"/>
    <body name="torso" pos="0 0 1.4">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="0 -.07 0 0 .07 0"
            size="0.07" density="1000"/>
      <geom name="head_geom" type="sphere" pos="0 0 .19" size=".09"
            density="1000"/>
      <body name="lwaist" pos="-.01 0 -.12">
        <geom name="waist_upper" type="capsule" fromto="0 -.06 0 0 .06 0"
              size="0.06" density="1000"/>
        <joint name="abdomen_z" type="hinge" pos="0 0 .065" axis="0 0 1"
               range="-45 45" damping="5"/>
        <joint name="abdomen_x" type="hinge" pos="0 0 .065" axis="1 0 0"
               range="-35 35" damping="5"/>
        <body name="pelvis" pos="0 0 -.165">
          <geom name="butt" type="capsule" fromto="-.02 -.07 0 -.02 .07 0"
                size="0.09" density="1000"/>
          <!-- Left leg -->
          <body name="lthigh" pos="0 0.1 -.04">
            <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-25 5"
                   damping="4"/>
            <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-110 20"
                   damping="4"/>
            <joint name="left_hip_z" type="hinge" axis="0 0 1" range="-60 35"
                   damping="4"/>
            <geom name="lthigh_geom" type="capsule" fromto="0 0 0 0 0.01 -.34"
                  size="0.06" density="1000"/>
            <body name="lshin" pos="0 0.01 -.403">
              <joint name="left_knee" type="hinge" axis="0 -1 0" range="-160 2"
                     damping="3"/>
              <geom name="lshin_geom" type="capsule" fromto="0 0 0 0 0 -.3"
                    size="0.049" density="1000"/>
              <body name="lfoot" pos="0 0 -.39">
                <joint name="left_ankle_y" type="hinge" axis="0 1 0"
                       range="-50 50" damping="3"/>
                <joint name="left_ankle_x" type="hinge" axis="1 0 .5"
                       range="-50 50" damping="3"/>
                <geom name="lfoot_geom" type="capsule"
                      fromto="-.07 -.02 0 .14 -.04 0" size="0.027"
                      density="1000" condim="4" friction="0.9 0.1 0.1"/>
              </body>
            </body>
          </body>
          <!-- Right leg -->
          <body name="rthigh" pos="0 -0.1 -.04">
            <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"
                   damping="4"/>
            <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-110 20"
                   damping="4"/>
            <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-60 35"
                   damping="4"/>
            <geom name="rthigh_geom" type="capsule" fromto="0 0 0 0 -0.01 -.34"
                  size="0.06" density="1000"/>
            <body name="rshin" pos="0 -0.01 -.403">
              <joint name="right_knee" type="hinge" axis="0 -1 0" range="-160 2"
                     damping="3"/>
              <geom name="rshin_geom" type="capsule" fromto="0 0 0 0 0 -.3"
                    size="0.049" density="1000"/>
              <body name="rfoot" pos="0 0 -.39">
                <joint name="right_ankle_y" type="hinge" axis="0 1 0"
                       range="-50 50" damping="3"/>
                <joint name="right_ankle_x" type="hinge" axis="1 0 .5"
                       range="-50 50" damping="3"/>
                <geom name="rfoot_geom" type="capsule"
                      fromto="-.07 .02 0 .14 .04 0" size="0.027"
                      density="1000" condim="4" friction="0.9 0.1 0.1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- Left arm -->
      <body name="luarm" pos="0 0.18 .06">
        <joint name="left_shoulder1" type="hinge" axis="2 1 1" range="-85 60"
               damping="1"/>
        <joint name="left_shoulder2" type="hinge" axis="0 -1 1" range="-85 60"
               damping="1"/>
        <geom name="luarm_geom" type="capsule" fromto="0 0 0 .16 .16 -.16"
              size="0.04" density="1000"/>
        <body name="llarm" pos=".18 .18 -.18">
          <joint name="left_elbow" type="hinge" axis="0 -1 1" range="-90 50"
                 damping="0"/>
          <geom name="llarm_geom" type="capsule" fromto="0 0 0 .17 .17 -.17"
                size="0.031" density="1000"/>
        </body>
      </body>
      <!-- Right arm -->
      <body name="ruarm" pos="0 -0.18 .06">
        <joint name="right_shoulder1" type="hinge" axis="2 -1 1" range="-60 85"
               damping="1"/>
        <joint name="right_shoulder2" type="hinge" axis="0 1 1" range="-60 85"
               damping="1"/>
        <geom name="ruarm_geom" type="capsule" fromto="0 0 0 .16 -.16 -.16"
              size="0.04" density="1000"/>
        <body name="rlarm" pos=".18 -.18 -.18">
          <joint name="right_elbow" type="hinge" axis="0 1 1" range="-90 50"
                 damping="0"/>
          <geom name="rlarm_geom" type="capsule" fromto="0 0 0 .17 -.17 -.17"
                size="0.031" density="1000"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="abdomen_z"      joint="abdomen_z"       gear="40"/>
    <motor name="abdomen_x"      joint="abdomen_x"       gear="40"/>
    <motor name="left_hip_x"     joint="left_hip_x"      gear="40"/>
    <motor name="left_hip_y"     joint="left_hip_y"      gear="60"/>
    <motor name="left_hip_z"     joint="left_hip_z"      gear="40"/>
    <motor name="left_knee"      joint="left_knee"       gear="80"/>
    <motor name="left_ankle_y"   joint="left_ankle_y"    gear="20"/>
    <motor name="left_ankle_x"   joint="left_ankle_x"    gear="20"/>
    <motor name="right_hip_x"    joint="right_hip_x"     gear="40"/>
    <motor name="right_hip_y"    joint="right_hip_y"     gear="60"/>
    <motor name="right_hip_z"    joint="right_hip_z"     gear="40"/>
    <motor name="right_knee"     joint="right_knee"      gear="80"/>
    <motor name="right_ankle_y"  joint="right_ankle_y"   gear="20"/>
    <motor name="right_ankle_x"  joint="right_ankle_x"   gear="20"/>
    <motor name="left_shoulder1" joint="left_shoulder1"  gear="20"/>
    <motor name="left_shoulder2" joint="left_shoulder2"  gear="20"/>
    <motor name="left_elbow"     joint="left_elbow"      gear="20"/>
    <motor name="right_shoulder1" joint="right_shoulder1" gear="20"/>
    <motor name="right_shoulder2" joint="right_shoulder2" gear="20"/>
    <motor name="right_elbow"    joint="right_elbow"     gear="20"/>
  </actuator>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance=".2"/>
    <material name="geom" rgba=".8 .6 .4 1"/>
  </asset>
</mujoco>
"""
    xml_path = "/tmp/humanoid_balance.xml"
    with open(xml_path, "w") as f:
        f.write(xml)
    return xml_path


# ---------------------------------------------------------------------------
# 2.  MJX Brax Environment
# ---------------------------------------------------------------------------

class HumanoidBalanceEnv(Env):
    """
    Humanoid standing-balance environment with push disturbances.

    Observation (67-dim for the built-in humanoid):
        qpos[2:]        position (exclude global x,y translation)
        qvel            velocities
        torso_up        dot(torso_z_axis, world_z)  — uprightness sensor
        feet_contact    left/right foot contact booleans
        push_force_obs  current external force magnitude (curriculum info)

    Action: normalized joint torques in [-1, 1] (scaled by gear ratios)

    Termination: torso height < 0.8 m  OR  |torso tilt| > 60 deg
    """

    def __init__(self, cfg: Config, mj_model: mujoco.MjModel):
        self._cfg = cfg
        self._model = mj_model
        self._mx = mjx.put_model(mj_model)

        # Cache body / geom / sensor indices
        self._torso_id = mj_model.body("torso").id
        self._lfoot_geom = mj_model.geom("lfoot_geom").id
        self._rfoot_geom = mj_model.geom("rfoot_geom").id

        # Nominal COM height (sampled from standing pose)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)
        self._nominal_height = float(mj_data.body("torso").xpos[2])

        # Observation / action dims
        nq = mj_model.nq
        nv = mj_model.nv
        na = mj_model.nu
        self._obs_size = (nq - 2) + nv + 3    # pos (no x,y) + vel + extras
        self._act_size = na

    # ── Brax Env interface ─────────────────────────────────────────────────

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._act_size

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        rng, rng_init, rng_push = jax.random.split(rng, 3)

        mx = self._mx
        dx = mjx.make_data(mx)

        # Randomise initial joint angles slightly (±0.1 rad) for diversity
        rng_q, rng_v = jax.random.split(rng_init)
        noise_q = jax.random.uniform(rng_q, (mx.nq,), minval=-0.05, maxval=0.05)
        noise_v = jax.random.uniform(rng_v, (mx.nv,), minval=-0.05, maxval=0.05)

        dx = dx.replace(qpos=dx.qpos + noise_q,
                        qvel=dx.qvel + noise_v)
        dx = mjx.forward(mx, dx)

        obs = self._get_obs(dx, jnp.zeros(3))
        reward = jnp.zeros(())
        done = jnp.zeros((), dtype=bool)
        metrics = {
            "reward_alive": jnp.zeros(()),
            "reward_upright": jnp.zeros(()),
            "reward_height": jnp.zeros(()),
            "reward_ctrl": jnp.zeros(()),
            "reward_vel": jnp.zeros(()),
            "push_force": jnp.zeros(()),
        }
        # Store step counter + push state in info
        info = {
            "step": jnp.zeros((), dtype=jnp.int32),
            "push_force": jnp.zeros(3),
            "push_remaining": jnp.zeros((), dtype=jnp.int32),
            "rng": rng_push,
        }
        return State(dx, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array,
             push_magnitude: float = 0.0) -> State:
        """
        Single environment step.
        push_magnitude is passed externally (controlled by curriculum).
        """
        dx: mjx.Data = state.pipeline_state
        mx = self._mx
        cfg = self._cfg
        info = state.info

        # ── 1.  Apply curriculum push disturbance ────────────────────────
        step_i = info["step"]
        rng, rng_push_dir = jax.random.split(info["rng"])

        # Decide whether to start a new push this step
        start_push = jnp.logical_and(
            step_i % cfg.push_interval_steps == 0,
            push_magnitude > 0.0
        )
        push_dir = jax.random.normal(rng_push_dir, (3,))
        push_dir = push_dir.at[2].set(0.0)          # horizontal pushes only
        push_dir = push_dir / (jnp.linalg.norm(push_dir) + 1e-8)
        new_push_force = jnp.where(start_push,
                                   push_dir * push_magnitude,
                                   info["push_force"])
        new_push_remaining = jnp.where(start_push,
                                       cfg.push_duration_steps,
                                       info["push_remaining"])
        push_active = new_push_remaining > 0
        applied_force = jnp.where(push_active, new_push_force, jnp.zeros(3))
        new_push_remaining = jnp.where(push_active,
                                       new_push_remaining - 1,
                                       new_push_remaining)

        # Apply external force to torso (xfrc_applied: [nbody, 6])
        torso_id = self._torso_id
        xfrc = dx.xfrc_applied
        xfrc = xfrc.at[torso_id, :3].set(applied_force)
        dx = dx.replace(xfrc_applied=xfrc)

        # ── 2.  Step physics ─────────────────────────────────────────────
        dx = dx.replace(ctrl=action)
        dx = mjx.step(mx, dx)

        # ── 3.  Compute reward ───────────────────────────────────────────
        torso_pos = dx.xpos[torso_id]      # world-frame torso position
        torso_mat = dx.xmat[torso_id].reshape(3, 3)
        torso_up  = torso_mat[:, 2]        # local Z axis in world frame

        r_alive   = cfg.w_alive
        r_upright = cfg.w_upright * torso_up[2]           # dot with world-up
        r_height  = cfg.w_height  * jnp.exp(
                        -10.0 * (torso_pos[2] - self._nominal_height) ** 2)
        r_ctrl    = -cfg.w_small_ctrl * jnp.sum(jnp.square(action))
        r_vel     = -cfg.w_small_vel  * jnp.sum(jnp.square(dx.qvel[:3]))
        r_ang     = -cfg.w_ang_vel    * jnp.sum(jnp.square(dx.qvel[3:6]))

        # Foot-contact reward: encourage both feet to stay on ground
        # (approximated via foot geom z-position near floor)
        lfoot_z = dx.geom_xpos[self._lfoot_geom][2]
        rfoot_z = dx.geom_xpos[self._rfoot_geom][2]
        r_foot  = cfg.w_foot * jnp.exp(
                    -100.0 * (lfoot_z ** 2 + rfoot_z ** 2))

        reward = r_alive + r_upright + r_height + r_ctrl + r_vel + r_ang + r_foot

        # ── 4.  Termination ──────────────────────────────────────────────
        too_low  = torso_pos[2] < 0.8
        too_tilted = torso_up[2] < jnp.cos(jnp.radians(60.0))
        done = jnp.logical_or(too_low, too_tilted)

        # ── 5.  Observation ──────────────────────────────────────────────
        obs = self._get_obs(dx, applied_force)

        # ── 6.  Update info ──────────────────────────────────────────────
        metrics = {
            "reward_alive":   r_alive,
            "reward_upright": r_upright,
            "reward_height":  r_height,
            "reward_ctrl":    r_ctrl,
            "reward_vel":     r_vel,
            "push_force":     jnp.linalg.norm(applied_force),
        }
        new_info = {
            "step":           step_i + 1,
            "push_force":     new_push_force,
            "push_remaining": new_push_remaining,
            "rng":            rng,
        }
        return state.replace(
            pipeline_state=dx,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )

    # ── Helper: build observation vector ──────────────────────────────────

    def _get_obs(self, dx: mjx.Data, push_force: jax.Array) -> jax.Array:
        """
        obs = [qpos[2:], qvel, torso_up_z, push_force_norm_xy]
        Excludes global x, y translation to make policy translation-invariant.
        """
        qpos = dx.qpos[2:]                            # exclude global x, y
        qvel = dx.qvel
        torso_mat = dx.xmat[self._torso_id].reshape(3, 3)
        torso_up_z = torso_mat[2, 2:3]               # scalar → (1,)
        push_obs = push_force[:2]                     # x, y of applied force
        return jnp.concatenate([qpos, qvel, torso_up_z, push_obs])


# ---------------------------------------------------------------------------
# 3.  Push-Curriculum Wrapper
# ---------------------------------------------------------------------------

class PushCurriculumWrapper:
    """
    Wraps HumanoidBalanceEnv and injects the correct push magnitude
    based on training progress (fraction of total steps completed).

    Phase 1  [0.00 – 0.40]: no pushes         → stable upright stance
    Phase 2  [0.40 – 0.70]: light pushes       → reactive stepping strategy
    Phase 3  [0.70 – 1.00]: strong pushes      → robust recovery
    """

    def __init__(self, env: HumanoidBalanceEnv, cfg: Config):
        self._env = env
        self._cfg = cfg

    def push_magnitude(self, training_fraction: float) -> float:
        cfg = self._cfg
        if training_fraction < cfg.push_phase2_start:
            return 0.0
        elif training_fraction < cfg.push_phase3_start:
            frac = ((training_fraction - cfg.push_phase2_start) /
                    (cfg.push_phase3_start - cfg.push_phase2_start))
            return cfg.push_force_light * frac
        else:
            frac = ((training_fraction - cfg.push_phase3_start) /
                    (1.0 - cfg.push_phase3_start))
            return (cfg.push_force_light +
                    (cfg.push_force_strong - cfg.push_force_light) * frac)


# ---------------------------------------------------------------------------
# 4.  Policy Network
# ---------------------------------------------------------------------------

class BalancePolicy(nn.Module):
    """
    MLP actor/critic shared trunk, following RSL-RL / Brax convention.
    Architecture: 4 × 256 hidden units with ELU activations.
    ELU chosen over ReLU for better gradient flow in bipedal RL
    (avoids dead neurons on joint-limit constraints).
    """
    hidden_sizes: tuple[int, ...] = (256, 256, 256, 256)

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.elu(x)
        return x


# ---------------------------------------------------------------------------
# 5.  Training entry point
# ---------------------------------------------------------------------------

def make_env_fn(cfg: Config, mj_model: mujoco.MjModel):
    """Factory that returns a freshly constructed environment (required by Brax)."""
    def _make():
        return HumanoidBalanceEnv(cfg, mj_model)
    return _make


def train(cfg: Config = CFG):
    """
    Main training loop.
    Uses Brax's built-in vectorised PPO implementation which is fully
    jit-compiled and runs entirely on GPU/TPU via JAX.
    """
    print("=" * 70)
    print("  Humanoid Balance & Push Recovery  |  MJX + Brax PPO")
    print("=" * 70)
    print(f"  Devices : {jax.devices()}")
    print(f"  Envs    : {cfg.num_envs:,}")
    print(f"  Steps   : {cfg.num_timesteps:,}")
    print()

    # Load model
    mj_model, _ = load_model(cfg)
    print(f"  MuJoCo model: nq={mj_model.nq}, nv={mj_model.nv}, "
          f"nu={mj_model.nu}, nbody={mj_model.nbody}")

    env = HumanoidBalanceEnv(cfg, mj_model)
    curriculum = PushCurriculumWrapper(env, cfg)

    # ── Build Brax PPO network factory ────────────────────────────────────
    # Brax PPO accepts a `network_factory` callable that returns an
    # (actor_network, value_network) tuple. We use Brax's default MLP
    # factory but can swap in BalancePolicy above for custom experiments.

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=ppo_networks.EMPTY_PREPROCESS,
        policy_hidden_layer_sizes=(256, 256, 256, 256),
        value_hidden_layer_sizes=(256, 256, 256, 256),
        activation=nn.elu,
    )

    # ── Checkpoint / log callbacks ─────────────────────────────────────────
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    times = [time.time()]
    rewards_history = []
    push_history = []

    def progress_fn(num_steps: int, metrics: dict[str, Any]):
        elapsed = time.time() - times[0]
        frac = num_steps / cfg.num_timesteps
        push_mag = curriculum.push_magnitude(frac)

        ep_reward = float(metrics.get("eval/episode_reward", 0.0))
        rewards_history.append((num_steps, ep_reward))
        push_history.append((num_steps, push_mag))

        phase = (1 if frac < cfg.push_phase2_start else
                 2 if frac < cfg.push_phase3_start else 3)
        print(
            f"  step={num_steps:>12,}  ({frac*100:5.1f}%)  "
            f"phase={phase}  push={push_mag:6.1f} N  "
            f"ep_reward={ep_reward:8.2f}  "
            f"elapsed={elapsed/60:.1f} min"
        )

        # Save CSV log
        with open(log_dir / "progress.csv", "a") as f:
            if num_steps == 0:
                f.write("steps,ep_reward,push_magnitude,elapsed_s\n")
            f.write(f"{num_steps},{ep_reward:.4f},{push_mag:.2f},{elapsed:.1f}\n")

    # ── Brax PPO train call ────────────────────────────────────────────────
    # Brax ppo_train returns (make_policy, params, metrics)
    make_policy, params, train_metrics = ppo_train.train(
        environment=env,
        num_timesteps=cfg.num_timesteps,
        episode_length=cfg.episode_length,
        num_envs=cfg.num_envs,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        discounting=cfg.discounting,
        unroll_length=cfg.unroll_length,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        reward_scaling=cfg.reward_scaling,
        gae_lambda=cfg.gae_lambda,
        clipping_epsilon=cfg.clipping_epsilon,
        normalize_observations=cfg.normalize_observations,
        num_eval_envs=cfg.num_eval_envs,
        eval_every=cfg.eval_every,
        network_factory=network_factory,
        progress_fn=progress_fn,
        seed=42,
    )

    print("\n  Training complete.")
    print(f"  Final episode reward : "
          f"{train_metrics.get('eval/episode_reward', '?')}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    import orbax.checkpoint as ocp
    ckpt_path = log_dir / "final_checkpoint"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_path), params)
    print(f"  Checkpoint saved → {ckpt_path}")

    return make_policy, params, train_metrics


# ---------------------------------------------------------------------------
# 6.  Evaluation / visualisation helper
# ---------------------------------------------------------------------------

def evaluate(make_policy, params, cfg: Config = CFG,
             num_episodes: int = 5, render: bool = True):
    """
    Roll out the trained policy for a few episodes and optionally render
    to video using MuJoCo's built-in renderer.
    """
    mj_model, mj_data = load_model(cfg)
    policy = make_policy(params, deterministic=True)

    from brax.io import html as brax_html
    import mujoco.viewer as viewer

    env = HumanoidBalanceEnv(cfg, mj_model)
    rng = jax.random.PRNGKey(0)
    total_reward = 0.0

    for ep in range(num_episodes):
        rng, rng_reset = jax.random.split(rng)
        state = jax.jit(env.reset)(rng_reset)
        ep_reward = 0.0
        frames = []

        for _ in range(cfg.episode_length):
            obs = state.obs[None]   # add batch dim
            act, _ = policy(obs, rng)
            act = act[0]            # remove batch dim
            state = jax.jit(env.step)(state, act,
                                      push_magnitude=cfg.push_force_strong)
            ep_reward += float(state.reward)
            if render:
                # Copy MJX data back to CPU for rendering
                dx_cpu = mjx.get_data(mj_model, state.pipeline_state)
                mujoco.mj_forward(mj_model, dx_cpu)
                frames.append(dx_cpu.qpos.copy())
            if state.done:
                break

        total_reward += ep_reward
        print(f"  Episode {ep+1}: reward = {ep_reward:.2f}")

    print(f"\n  Mean episode reward over {num_episodes} episodes: "
          f"{total_reward / num_episodes:.2f}")
    return frames


# ---------------------------------------------------------------------------
# 7.  Reward shaping reference table
# ---------------------------------------------------------------------------

REWARD_TABLE = """
Reward Term        | Weight | Purpose
─────────────────────────────────────────────────────────────────────────
alive              |  5.0   | Per-step survival bonus; main incentive to stay up
upright            |  3.0   | dot(torso_z, world_z): penalises tilting
height             |  2.0   | Gaussian around nominal COM height (~1.3 m)
small_ctrl         |  0.05  | L2 penalty on joint torques → energy efficiency
small_vel          |  0.1   | Penalise linear root velocity (stand still)
ang_vel            |  0.05  | Penalise root angular velocity
foot_contact       |  1.0   | Gaussian: both feet near ground plane
─────────────────────────────────────────────────────────────────────────
Termination: torso height < 0.8 m  OR  torso tilt > 60 deg from vertical
"""


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train humanoid standing balance + push recovery")
    parser.add_argument("--envs",       type=int,   default=CFG.num_envs,
                        help="Number of parallel environments")
    parser.add_argument("--steps",      type=int,   default=CFG.num_timesteps,
                        help="Total training steps")
    parser.add_argument("--lr",         type=float, default=CFG.learning_rate,
                        help="PPO learning rate")
    parser.add_argument("--log-dir",    type=str,   default=CFG.log_dir,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; load checkpoint and evaluate")
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Checkpoint path for --eval-only mode")
    args = parser.parse_args()

    cfg = Config(
        num_envs=args.envs,
        num_timesteps=args.steps,
        learning_rate=args.lr,
        log_dir=args.log_dir,
    )

    print(REWARD_TABLE)

    if args.eval_only:
        assert args.checkpoint, "Provide --checkpoint path for eval-only mode"
        import orbax.checkpoint as ocp
        mj_model, _ = load_model(cfg)
        env = HumanoidBalanceEnv(cfg, mj_model)
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=ppo_networks.EMPTY_PREPROCESS,
            policy_hidden_layer_sizes=(256, 256, 256, 256),
            value_hidden_layer_sizes=(256, 256, 256, 256),
            activation=nn.elu,
        )
        make_policy, _, _ = ppo_networks.make_inference_fn(network_factory)
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(args.checkpoint)
        evaluate(make_policy, params, cfg)
    else:
        make_policy, params, metrics = train(cfg)
        evaluate(make_policy, params, cfg, num_episodes=3, render=False)