import os
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import humanoid_balance_v4 as hb4

os.makedirs("assets/media", exist_ok=True)

def render_v4_demo():
    print("Rendering v4 demo...")
    ckpt_path = "runs/humanoid_balance_v4/best/best_model.zip"
    vec_path = "runs/humanoid_balance_v4/best/best_vecnormalize.pkl"
    if not os.path.exists(ckpt_path):
        ckpts = sorted(list(Path("runs/humanoid_balance_v4/ckpts").glob("ppo_*_steps.zip")))
        if ckpts:
            ckpt_path = str(ckpts[-1])
            vec_path = ckpt_path.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")

    def make_env():
        return hb4.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=250.0)

    env = DummyVecEnv([make_env])
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(ckpt_path, env=env)

    obs = env.reset()
    frames = []
    # simulate 200 frames
    for t in range(200):
        unwrapped = env.envs[0]
        if t == 60 or t == 130:
            unwrapped.trigger_push()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if done[0]:
            obs = env.reset()

    env.close()

    # Save representative screenshots
    imageio.imwrite("assets/media/v4_standing_hero.png", frames[40])
    imageio.imwrite("assets/media/v4_push_recovery.png", frames[85])
    imageio.imwrite("assets/media/v4_settled_stance.png", frames[150])

    # Save animated GIF
    subsampled = frames[::2]
    imageio.mimsave("assets/media/v4_recovery_demo.gif", subsampled, fps=25, loop=0)
    print("Saved v4 demo gif and screenshots!")

def render_v1_demo():
    print("Rendering v1 baseline fall demo...")
    ckpts = sorted(list(Path("runs/humanoid_balance/ckpts").glob("ppo_*_steps.zip")))
    if not ckpts:
        return
    ckpt_path = str(ckpts[0])

    import humanoid_balance_gymnasium as hb1

    def make_env():
        return hb1.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=50.0)

    env = DummyVecEnv([make_env])
    vec_path = ckpt_path.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False

    try:
        model = PPO.load(ckpt_path, env=env)
        obs = env.reset()
        frames = []
        for t in range(120):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frames.append(env.render())
            if done[0]:
                break
        env.close()
        imageio.imwrite("assets/media/v1_fall.png", frames[-1] if frames else np.zeros((480,480,3), dtype=np.uint8))
        if len(frames) > 5:
            imageio.mimsave("assets/media/v1_fall_demo.gif", frames[::2], fps=20, loop=0)
        print("Saved v1 demo!")
    except Exception as e:
        print("v1 render note:", e)

if __name__ == "__main__":
    render_v4_demo()
    try:
        render_v1_demo()
    except Exception as e:
        print("V1 render skipped:", e)
