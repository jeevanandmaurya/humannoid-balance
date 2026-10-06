"""
Sim2Sim Test: watch the MuJoCo-trained Humanoid balance policy run in PyBullet
================================================================================
Same checkpoints, different physics engine. This does NOT retrain or fine-tune
anything -- it takes a PPO checkpoint trained in humanoid_balance_gymnasium.py
(MuJoCo / Gymnasium Humanoid-v5) and drives the *identical* MJCF robot model
inside PyBullet, to see how much the policy's balance/recovery skill survives
the swap in contact solver, integrator, and actuator dynamics.

Why this is legitimate sim2sim (not just "a different humanoid"):
  - We load the SAME humanoid.xml asset Gymnasium's Humanoid-v5 uses, via
    PyBullet's MJCF importer, so link masses, geometry and joint layout match.
  - To keep the *observation encoding* apples-to-apples with training, we keep
    a "shadow" MuJoCo model+data around (never integrated/stepped) and copy
    PyBullet's simulated qpos/qvel into it every frame, then call
    mujoco.mj_forward() + the real Gymnasium Humanoid-v5 _get_obs(). That
    reproduces the exact 348-dim observation formula (cinert/cvel/
    qfrc_actuator/cfrc_ext included) the policy was trained on, just fed with
    PyBullet-simulated numbers instead of MuJoCo-simulated ones.
  - The push disturbance, curriculum magnitude, and fallen/height thresholds
    are reused from the training script so "watch mode" behaves the same way.

What will NOT match, on purpose -- that's the whole point of the test:
  - Contact/friction solver, joint damping/armature handling, and the
    actuator model differ between MuJoCo and PyBullet even with the same XML,
    so torques that balanced the robot in MuJoCo may not balance it here.
  - qfrc_actuator / cfrc_ext in the shadow obs are *approximations*: they're
    recomputed by MuJoCo from the current state+ctrl via mj_forward, not
    measured from PyBullet's own solver, since PyBullet has no equivalent
    quantities. Treat degradation partly attributable to this as expected
    sim2sim observation noise, not just dynamics mismatch.

Run:
    python humanoid_sim2sim_pybullet.py
    python humanoid_sim2sim_pybullet.py --checkpoint runs/humanoid_balance_v2/best
    python humanoid_sim2sim_pybullet.py --checkpoint runs/humanoid_balance_v2/ckpts/ppo_10000000_steps.zip
    python humanoid_sim2sim_pybullet.py --push 100 --episodes 5
    python humanoid_sim2sim_pybullet.py --no-gui   # headless, just prints stats

Install:
    pip install pybullet
    (gymnasium[mujoco], stable-baselines3, mujoco already required by the
     training script)
"""

import argparse
import pickle
import re
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

# Reuse constants / curriculum from the training script so watch behavior
# (push schedule, thresholds, checkpoint layout) stays consistent. Falls back
# to local copies if the training file has a different name / isn't importable.
try:
    import humanoid_balance_gymnasium as base
    LOG_DIR        = base.LOG_DIR
    PUSH_LIGHT     = base.PUSH_LIGHT
    PUSH_STRONG    = base.PUSH_STRONG
    PUSH_PHASE2    = base.PUSH_PHASE2
    PUSH_PHASE3    = base.PUSH_PHASE3
    PUSH_INTERVAL  = base.PUSH_INTERVAL
    PUSH_DURATION  = base.PUSH_DURATION
    TOTAL_STEPS    = base.TOTAL_STEPS
    EPISODE_LENGTH = base.EPISODE_LENGTH
    curriculum_push = base.curriculum_push
except ImportError:
    print("[warn] could not import humanoid_balance_gymnasium.py, "
          "using local fallback constants.")
    LOG_DIR        = Path("./runs/humanoid_balance_v2")
    PUSH_LIGHT     = 50.0
    PUSH_STRONG    = 150.0
    PUSH_PHASE2    = 0.10
    PUSH_PHASE3    = 0.30
    PUSH_INTERVAL  = 200
    PUSH_DURATION  = 5
    TOTAL_STEPS    = 50_000_000
    EPISODE_LENGTH = 1000

    def curriculum_push(step: int) -> float:
        frac = min(max(step / TOTAL_STEPS, 0.0), 1.0)
        if frac < PUSH_PHASE2:
            return 0.0
        if frac < PUSH_PHASE3:
            return PUSH_LIGHT * (frac - PUSH_PHASE2) / (PUSH_PHASE3 - PUSH_PHASE2)
        return PUSH_LIGHT + (PUSH_STRONG - PUSH_LIGHT) * (
            (frac - PUSH_PHASE3) / (1.0 - PUSH_PHASE3))


FALLEN_HEIGHT_M   = 0.8
FALLEN_UPRIGHT_DZ = np.cos(np.deg2rad(60.0))
SIM_HZ            = 500          # PyBullet inner physics rate
CONTROL_HZ        = 1.0 / 0.015  # Gymnasium Humanoid-v5 default dt (frame_skip*timestep)


# ---------------------------------------------------------------------------
# Shadow MuJoCo model: physics is never stepped here, only used to compute
# the exact same observation encoding _get_obs() produces during training.
# ---------------------------------------------------------------------------

class ShadowMuJoCoObs:
    def __init__(self):
        self._env = gym.make(
            "Humanoid-v5",
            forward_reward_weight=0.0, ctrl_cost_weight=0.0,
            contact_cost_weight=0.0, healthy_reward=0.0,
            terminate_when_unhealthy=False,
        )
        obs0, _ = self._env.reset()
        self.model = self._env.unwrapped.model
        self.data = self._env.unwrapped.data
        self.mjcf_path = self._env.unwrapped.fullpath
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.base_obs_dim = obs0.shape[0]

        self.torso_id = self.model.body("torso").id
        self.right_foot_geom = self.model.geom("right_foot").id
        self.left_foot_geom = self.model.geom("left_foot").id

        # Actuator order == action-vector order. Map each actuator to the
        # joint it drives so we can command matching torques in PyBullet.
        self.actuator_joint_names = []
        self.actuator_gear = self.model.actuator_gear[:, 0].copy()
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            self.actuator_joint_names.append(self.model.joint(joint_id).name)

        # Non-free hinge joints: name -> (qposadr, qveladr), in mj order.
        self.hinge_joint_qpos_adr = {}
        self.hinge_joint_qvel_adr = {}
        for name in self.actuator_joint_names:
            j = self.model.joint(name)
            self.hinge_joint_qpos_adr[name] = int(j.qposadr[0])
            # qveladr was renamed to dofadr in newer MuJoCo versions
            if hasattr(j, 'dofadr'):
                self.hinge_joint_qvel_adr[name] = int(j.dofadr[0])
            else:
                self.hinge_joint_qvel_adr[name] = int(j.qveladr[0])

    def sync_and_get_obs(self, qpos, qvel, ctrl):
        """Push a PyBullet-derived state into the shadow model, recompute
        derived quantities with mj_forward, and return the training-format
        observation (WITHOUT the push_force/prev_action augmentation)."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        return self._env.unwrapped._get_obs()

    def fresh_init_state(self):
        obs, _ = self._env.reset()
        return self.data.qpos.copy(), self.data.qvel.copy()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# PyBullet world built from the SAME MJCF asset
# ---------------------------------------------------------------------------

class PyBulletHumanoid:
    def __init__(self, shadow: ShadowMuJoCoObs, gui=True):
        self.shadow = shadow
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_HZ)
        self.plane = p.loadURDF("plane.urdf")

        flags = p.URDF_USE_SELF_COLLISION | p.MJCF_COLORS_FROM_FILE
        bodies = p.loadMJCF(shadow.mjcf_path, flags=flags)
        if not bodies:
            raise RuntimeError(f"PyBullet failed to load MJCF: {shadow.mjcf_path}")
        # humanoid.xml defines exactly one free-floating robot body (the
        # ground plane is added separately above), so take the first result.
        self.body = bodies[0]

        # name -> pybullet joint index, and disable default velocity motors
        # so we can drive pure torque control (mirrors MuJoCo's motor actuators).
        self.joint_index = {}
        for i in range(p.getNumJoints(self.body)):
            info = p.getJointInfo(self.body, i)
            name = info[1].decode("utf-8")
            self.joint_index[name] = i
            p.setJointMotorControl2(self.body, i, controlMode=p.VELOCITY_CONTROL,
                                     force=0)

        missing = [n for n in shadow.actuator_joint_names if n not in self.joint_index]
        if missing:
            raise RuntimeError(
                f"PyBullet MJCF import is missing joints the policy expects: "
                f"{missing}. The two engines' models diverged -- check that "
                f"PyBullet's MJCF importer parsed humanoid.xml fully.")
        self.pb_joint_order = [self.joint_index[n] for n in shadow.actuator_joint_names]

        self.reset_state(*shadow.fresh_init_state())

    # -- state transfer helpers -------------------------------------------------

    def reset_state(self, qpos, qvel):
        pos = qpos[0:3]
        quat_wxyz = qpos[3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        p.resetBasePositionAndOrientation(self.body, pos, quat_xyzw)
        lin_vel_world = qvel[0:3]
        ang_vel_local = qvel[3:6]
        rot = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
        ang_vel_world = rot @ ang_vel_local
        p.resetBaseVelocity(self.body, lin_vel_world.tolist(), ang_vel_world.tolist())
        for name in self.shadow.actuator_joint_names:
            qadr = self.shadow.hinge_joint_qpos_adr[name]
            vadr = self.shadow.hinge_joint_qvel_adr[name]
            p.resetJointState(self.body, self.joint_index[name],
                               targetValue=qpos[qadr], targetVelocity=qvel[vadr])

    def read_mujoco_style_state(self):
        """Read PyBullet's current state back out in MuJoCo's qpos/qvel layout."""
        qpos = np.zeros_like(self.shadow.init_qpos)
        qvel = np.zeros_like(self.shadow.init_qvel)

        pos, quat_xyzw = p.getBasePositionAndOrientation(self.body)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(self.body)
        rot = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
        ang_vel_local = rot.T @ np.array(ang_vel_world)

        qpos[0:3] = pos
        qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        qvel[0:3] = lin_vel_world
        qvel[3:6] = ang_vel_local

        for name in self.shadow.actuator_joint_names:
            j_pos, j_vel, _, _ = p.getJointState(self.body, self.joint_index[name])
            qpos[self.shadow.hinge_joint_qpos_adr[name]] = j_pos
            qvel[self.shadow.hinge_joint_qvel_adr[name]] = j_vel
        return qpos, qvel, pos, rot

    # -- control ------------------------------------------------------------

    def apply_action(self, action):
        torques = action * self.shadow.actuator_gear
        p.setJointMotorControlArray(
            self.body, self.pb_joint_order,
            controlMode=p.TORQUE_CONTROL, forces=torques.tolist())

    def apply_push(self, force_xy, torso_world_pos):
        if abs(force_xy[0]) < 1e-9 and abs(force_xy[1]) < 1e-9:
            return
        p.applyExternalForce(
            self.body, -1, [force_xy[0], force_xy[1], 0.0],
            torso_world_pos, p.WORLD_FRAME)

    def close(self):
        p.disconnect(self.client)


# ---------------------------------------------------------------------------
# Checkpoint / normalization loading (mirrors the training script's watch())
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint):
    if checkpoint:
        requested = Path(checkpoint)
    else:
        ckpt_dir = LOG_DIR / "ckpts"
        saved = list(ckpt_dir.glob("ppo_*_steps.zip"))
        if saved:
            def step_number(path):
                m = re.search(r"ppo_(\d+)_steps\.zip$", path.name)
                return int(m.group(1)) if m else -1
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
        model_path = requested if requested.suffix == ".zip" else requested.with_suffix(".zip")

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    norm_candidates = [
        model_path.with_name(model_path.stem + "_vecnormalize.pkl"),
        LOG_DIR / "vecnormalize.pkl",
    ]
    m = re.match(r"ppo_(\d+)_steps$", model_path.stem)
    if m:
        norm_candidates.insert(1, model_path.with_name(
            f"ppo_vecnormalize_{m.group(1)}_steps.pkl"))
    if model_path.parent.name == "best":
        norm_candidates.insert(0, model_path.parent / "best_model_vecnormalize.pkl")
    norm_path = next((p for p in norm_candidates if p.exists()), None)

    step_match = re.match(r"ppo_(\d+)_steps\.zip$", model_path.name)
    push_mag = curriculum_push(int(step_match.group(1))) if step_match else PUSH_STRONG
    return model_path, norm_path, push_mag


def load_obs_normalizer(norm_path):
    if norm_path is None:
        return None
    with open(norm_path, "rb") as f:
        vecnorm = pickle.load(f)
    return vecnorm.obs_rms, vecnorm.clip_obs, vecnorm.epsilon


def normalize_obs(obs, obs_rms, clip_obs, epsilon):
    if obs_rms is None:
        return obs
    normed = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
    return np.clip(normed, -clip_obs, clip_obs).astype(np.float32)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

class Hud:
    def __init__(self):
        self.text_id = None
        self.line_id = None

    def update(self, text, torso_pos, push_force):
        self.text_id = p.addUserDebugText(
            text, [torso_pos[0] - 0.6, torso_pos[1], torso_pos[2] + 1.0],
            textColorRGB=[1, 1, 1], textSize=1.3,
            replaceItemUniqueId=self.text_id if self.text_id is not None else -1)
        mag = float(np.linalg.norm(push_force))
        if mag > 1e-6:
            direction = np.array(push_force) / mag
            start = np.array(torso_pos) + direction * 0.15
            end = start + direction * (0.25 + min(mag / 500.0, 0.5))
            self.line_id = p.addUserDebugLine(
                start.tolist(), end.tolist(), [1, 0.05, 0.05], lineWidth=4,
                replaceItemUniqueId=self.line_id if self.line_id is not None else -1)
        elif self.line_id is not None:
            p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 0, 0],
                                replaceItemUniqueId=self.line_id)


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------

def run(checkpoint=None, episodes=3, continuous=False, push_override=None, gui=True):
    model_path, norm_path, curriculum_push_mag = resolve_checkpoint(checkpoint)
    push_mag = push_override if push_override is not None else curriculum_push_mag
    obs_rms, clip_obs, epsilon = load_obs_normalizer(norm_path)
    if obs_rms is None:
        print("[warn] no VecNormalize stats found -- feeding raw (unnormalized) "
              "observations to the policy. Behavior will likely look broken "
              "even before you blame the physics engine.")

    print(f"Sim2Sim (PyBullet) watching: {model_path}")
    print(f"Push magnitude: {push_mag:.1f} N   |   normalization: "
          f"{norm_path if norm_path else 'NONE'}")

    shadow = ShadowMuJoCoObs()
    world = PyBulletHumanoid(shadow, gui=gui)
    policy = PPO.load(str(model_path), device="cpu")

    hud = Hud()
    sub_steps = max(int(round(SIM_HZ / CONTROL_HZ)), 1)

    previous_action = np.zeros(shadow.model.nu, dtype=np.float32)
    push_force = np.zeros(3)
    push_remaining = 0
    step_count = 0
    episode = 0

    def trigger_push():
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle) * push_mag, np.sin(angle) * push_mag, 0.0])

    try:
        while continuous or episode < episodes:
            qpos, qvel = shadow.fresh_init_state()
            world.reset_state(qpos, qvel)
            previous_action[:] = 0.0
            push_force[:] = 0.0
            push_remaining = 0
            step_count = 0
            episode_start = time.perf_counter()
            print(f"\n[sim2sim] Episode {episode + 1} started (push={push_mag:.0f} N)")

            fallen = False
            while True:
                qpos, qvel, torso_pos, torso_rot = world.read_mujoco_style_state()
                torso_z = torso_pos[2]
                torso_up_z = torso_rot[2, 2]
                fallen = torso_z < FALLEN_HEIGHT_M or torso_up_z < FALLEN_UPRIGHT_DZ

                base_obs = shadow.sync_and_get_obs(qpos, qvel, previous_action)
                full_obs = np.concatenate(
                    [base_obs, push_force[:2], previous_action]).astype(np.float32)
                norm_obs = normalize_obs(full_obs, obs_rms, clip_obs, epsilon)

                action, _ = policy.predict(norm_obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)

                if push_mag > 0 and step_count > 0 and step_count % PUSH_INTERVAL == 0:
                    push_force = trigger_push()
                    push_remaining = PUSH_DURATION
                current_push = push_force if push_remaining > 0 else np.zeros(3)
                if push_remaining > 0:
                    push_remaining -= 1

                world.apply_action(action)
                world.apply_push(current_push[:2], torso_pos)
                for _ in range(sub_steps):
                    p.stepSimulation()
                if gui:
                    time.sleep(1.0 / CONTROL_HZ)

                previous_action = action.copy()
                step_count += 1

                if gui:
                    hud.update(
                        f"PYBULLET SIM2SIM | HEIGHT {torso_z:.2f}m | "
                        f"UPRIGHT {torso_up_z:.2f} | PUSH {push_mag:.0f}N | "
                        f"step {step_count}",
                        torso_pos, current_push)

                if fallen or step_count >= EPISODE_LENGTH:
                    break
                if gui and not p.isConnected():
                    return

            episode += 1
            elapsed = time.perf_counter() - episode_start
            result = "FELL" if fallen else "TIME LIMIT"
            print(f"[sim2sim] Episode {episode}: {result} after {step_count} steps "
                  f"({elapsed:.1f}s wall clock)")
    except KeyboardInterrupt:
        print("\n[sim2sim] stopped.")
    finally:
        shadow.close()
        world.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a .zip checkpoint or a run directory (defaults "
                          "to the latest checkpoint under LOG_DIR/ckpts).")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--continuous", action="store_true",
                     help="Keep looping episodes until Ctrl+C / window closed.")
    ap.add_argument("--push", type=float, default=None,
                     help="Override push force in N (default: curriculum value "
                          "for the loaded checkpoint's training step, or the "
                          "strong-push stress test for non-numbered checkpoints).")
    ap.add_argument("--no-gui", action="store_true",
                     help="Run headless (DIRECT mode), just print episode stats.")
    args = ap.parse_args()

    run(checkpoint=args.checkpoint, episodes=args.episodes,
        continuous=args.continuous, push_override=args.push, gui=not args.no_gui)