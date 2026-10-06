import os
import glob
import numpy as np
import gymnasium as gym
import imageio
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import humanoid_balance_gymnasium as hb1
import humanoid_balance_v3 as hb3
import humanoid_balance_v4 as hb4

os.makedirs("assets/media/versions", exist_ok=True)

def save_gif(frames, path, fps=30):
    """Save frames as a GIF."""
    if frames:
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"  Saved {path} ({len(frames)} frames)")

# ── Observation-compatible v1 wrapper (matches the 350-D vecnormalize checkpoint) ──
class V1CompatEnv(gym.Wrapper):
    """v1 checkpoints were trained with 350-D obs: base(348) + push_force(2), no action buffer."""
    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        super().__init__(env)
        base_dim = env.observation_space.shape[0]  # 348
        high = np.full(base_dim + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._push_force = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate((obs, self._push_force)).astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return np.concatenate((obs, self._push_force)).astype(np.float32), r, term, trunc, info


# ═══════════════════════════════════════════════
# 1. Free Fall GIF (untrained, zero actions)
# ═══════════════════════════════════════════════
def render_free_fall_gif():
    print("Rendering free fall GIF...")
    # terminate_when_unhealthy=False lets the sim keep running after the fall
    # so we capture the full collapse to the ground
    env = gym.make("Humanoid-v5", render_mode="rgb_array", terminate_when_unhealthy=False)
    env.reset()
    frames = []
    for _ in range(120):  # 120 frames @ 25fps = ~5s, enough to hit the ground fully
        _, _, done, _, _ = env.step(np.zeros(env.action_space.shape))
        frames.append(env.render())
    env.close()
    # hold the final collapsed frame for 30 extra frames so it's clear it's down
    if frames:
        frames += [frames[-1]] * 20
    save_gif(frames, "assets/media/versions/step0_free_fall.gif", fps=25)


# ═══════════════════════════════════════════════
# 2. v1 Final GIF (collapses — full fall to ground)
# ═══════════════════════════════════════════════
class V1CompatEnvNoTerminate(gym.Wrapper):
    """Like V1CompatEnv but wraps a non-terminating base env so we see the full fall."""
    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v5", render_mode=render_mode, terminate_when_unhealthy=False)
        super().__init__(env)
        base_dim = env.observation_space.shape[0]
        high = np.full(base_dim + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._push_force = np.zeros(2, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate((obs, self._push_force)).astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return np.concatenate((obs, self._push_force)).astype(np.float32), r, term, trunc, info


def render_v1_gif():
    print("Rendering v1 final GIF (full collapse)...")
    ckpts = sorted(glob.glob("runs/humanoid_balance/ckpts/ppo_*_steps.zip"))
    if not ckpts:
        print("  No v1 checkpoints found")
        return
    ckpt = ckpts[-1]
    vec = ckpt.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")

    env = DummyVecEnv([lambda: V1CompatEnvNoTerminate(render_mode="rgb_array")])
    if os.path.exists(vec):
        env = VecNormalize.load(vec, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(ckpt, env=env, device="cpu")
    obs = env.reset()
    frames = []
    # Run for 150 frames: policy acts first, then fall, then hold the ground state
    for _ in range(150):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        frames.append(env.render())
    env.close()
    # hold the final on-ground frame
    if frames:
        frames += [frames[-1]] * 20
    save_gif(frames, "assets/media/versions/v1_final.gif", fps=25)


# ═══════════════════════════════════════════════
# 3. v2 Final GIF (stable balance)
# ═══════════════════════════════════════════════
def render_v2_gif():
    print("Rendering v2 final GIF...")
    model_path = "runs/humanoid_balance_v2/final_model.zip"
    vec_path = "runs/humanoid_balance_v2/vecnormalize.pkl"
    if not os.path.exists(vec_path):
        vec_path = "runs/humanoid_balance_v2/best/best_model_vecnormalize.pkl"
        model_path = "runs/humanoid_balance_v2/best/best_model.zip"

    env = DummyVecEnv([lambda: hb1.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=80.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")
    obs = env.reset()
    frames = []
    for t in range(120):
        if t == 40:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()
    save_gif(frames, "assets/media/versions/v2_final.gif", fps=25)


# ═══════════════════════════════════════════════
# 4. v3 Best Final GIF (origin-locked, best model)
# ═══════════════════════════════════════════════
def render_v3_gif():
    print("Rendering v3 best GIF...")
    model_path = "runs/humanoid_balance_v3/final_model.zip"
    vec_path = "runs/humanoid_balance_v3/vecnormalize.pkl"
    if not os.path.exists(vec_path):
        vec_path = "runs/humanoid_balance_v3/best/best_model_vecnormalize.pkl"
        model_path = "runs/humanoid_balance_v3/best/best_model.zip"

    env = DummyVecEnv([lambda: hb3.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=150.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")
    obs = env.reset()
    frames = []
    for t in range(120):
        if t == 40:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()
    save_gif(frames, "assets/media/versions/v3_best.gif", fps=25)


# ═══════════════════════════════════════════════
# 5. v3 Push Recovery Demo GIF (for the main demo section)
# ═══════════════════════════════════════════════
def render_v3_demo_gif():
    print("Rendering v3 push recovery demo GIF...")
    model_path = "runs/humanoid_balance_v3/final_model.zip"
    vec_path = "runs/humanoid_balance_v3/vecnormalize.pkl"
    if not os.path.exists(vec_path):
        vec_path = "runs/humanoid_balance_v3/best/best_model_vecnormalize.pkl"
        model_path = "runs/humanoid_balance_v3/best/best_model.zip"

    env = DummyVecEnv([lambda: hb3.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=150.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")
    obs = env.reset()
    frames = []
    # Let it stand calmly first, then push hard
    for t in range(180):
        if t == 50:
            env.envs[0].push_magnitude = 200.0
            env.envs[0].trigger_push()
        if t == 120:
            env.envs[0].push_magnitude = 150.0
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()
    save_gif(frames, "assets/media/v3_recovery_demo.gif", fps=30)


# ═══════════════════════════════════════════════
# 6. v4 Experimental GIF
# ═══════════════════════════════════════════════
def render_v4_gif():
    print("Rendering v4 experimental GIF...")
    model_path = "runs/humanoid_balance_v4/best/best_model.zip"
    vec_path = "runs/humanoid_balance_v4/best/best_model_vecnormalize.pkl"

    env = DummyVecEnv([lambda: hb4.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=100.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")
    obs = env.reset()
    frames = []
    for t in range(120):
        if t == 40:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()
    save_gif(frames, "assets/media/versions/v4_experimental.gif", fps=25)


if __name__ == "__main__":
    render_free_fall_gif()
    render_v1_gif()
    render_v2_gif()
    render_v3_gif()
    render_v3_demo_gif()
    render_v4_gif()
    print("\nAll GIFs rendered!")
