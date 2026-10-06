import os
import glob
import numpy as np
import gymnasium as gym
import imageio
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import humanoid_balance_v3 as hb3
import humanoid_balance_v4 as hb4

os.makedirs("assets/media/versions", exist_ok=True)

class V1RawEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
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

def render_v1():
    print("Rendering v1 final...")
    ckpts = sorted(glob.glob("runs/humanoid_balance/ckpts/ppo_*_steps.zip"))
    if not ckpts:
        return
    ckpt = ckpts[-1]
    vec = ckpt.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    
    env = DummyVecEnv([lambda: V1RawEnv(render_mode="rgb_array")])
    if os.path.exists(vec):
        env = VecNormalize.load(vec, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(ckpt, env=env)
    obs = env.reset()
    frames = []
    for _ in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()
    if frames:
        imageio.imwrite("assets/media/versions/v1_final_fall.png", frames[-1])
        print("Saved assets/media/versions/v1_final_fall.png")

def render_v2():
    print("Rendering v2 final...")
    model_path = "runs/humanoid_balance_v2/final_model.zip"
    vec_path = "runs/humanoid_balance_v2/vecnormalize.pkl"
    import humanoid_balance_gymnasium as hb1
    env = DummyVecEnv([lambda: hb1.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=80.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    frames = []
    for t in range(80):
        if t == 30:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]: break
    env.close()
    if frames:
        imageio.imwrite("assets/media/versions/v2_final_balanced.png", frames[45])
        print("Saved assets/media/versions/v2_final_balanced.png")

def render_v3():
    print("Rendering v3 best final...")
    model_path = "runs/humanoid_balance_v3/final_model.zip"
    vec_path = "runs/humanoid_balance_v3/vecnormalize.pkl"
    env = DummyVecEnv([lambda: hb3.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=150.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    frames = []
    for t in range(90):
        if t == 30:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]: break
    env.close()
    if frames:
        imageio.imwrite("assets/media/versions/v3_best_final.png", frames[50])
        print("Saved assets/media/versions/v3_best_final.png")

def render_v4():
    print("Rendering v4 experimental...")
    model_path = "runs/humanoid_balance_v4/best/best_model.zip"
    vec_path = "runs/humanoid_balance_v4/best/best_model_vecnormalize.pkl"
    env = DummyVecEnv([lambda: hb4.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=100.0)])
    env = VecNormalize.load(vec_path, env)
    env.training = False
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    frames = []
    for t in range(80):
        if t == 30:
            env.envs[0].trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        frames.append(env.render())
        if done[0]: break
    env.close()
    if frames:
        imageio.imwrite("assets/media/versions/v4_experimental.png", frames[45])
        print("Saved assets/media/versions/v4_experimental.png")

if __name__ == "__main__":
    try: render_v1()
    except Exception as e: print("v1 error:", e)
    try: render_v2()
    except Exception as e: print("v2 error:", e)
    try: render_v3()
    except Exception as e: print("v3 error:", e)
    try: render_v4()
    except Exception as e: print("v4 error:", e)
    print("Render completed!")
