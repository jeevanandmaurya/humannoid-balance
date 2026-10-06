import os
import glob
import numpy as np
import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import humanoid_balance_gymnasium as hb1
import humanoid_balance_v3 as hb3
import humanoid_balance_v4 as hb4

os.makedirs("assets/media/versions", exist_ok=True)

def render_model(env_fn, model_path, vec_path, out_name, n_steps=120, push_at=None, push_mag=100.0):
    print(f"Rendering {out_name} from {model_path}...")
    env = DummyVecEnv([env_fn])
    if vec_path and os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False

    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        env.close()
        return

    obs = env.reset()
    frames = []
    for t in range(n_steps):
        if push_at and t == push_at:
            unwrapped = env.envs[0]
            if hasattr(unwrapped, 'trigger_push'):
                unwrapped.push_magnitude = push_mag
                unwrapped.trigger_push()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.render())
        if done[0]:
            break
    env.close()

    if frames:
        imageio.imwrite(f"assets/media/versions/{out_name}.png", frames[min(len(frames)-1, 45)])
        print(f"Saved assets/media/versions/{out_name}.png")

# 1. Untrained Free Fall (Baseline Step 0)
def render_free_fall():
    print("Rendering free fall...")
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env.reset()
    frames = []
    for _ in range(40):
        # zero action -> pure passive gravitational collapse
        _, _, done, _, _ = env.step(np.zeros(env.action_space.shape))
        frames.append(env.render())
        if done: break
    env.close()
    if frames:
        imageio.imwrite("assets/media/versions/step0_free_fall.png", frames[min(25, len(frames)-1)])
        print("Saved free fall image!")

# 2. v1 final (collapsed)
def render_v1_final():
    ckpts = sorted(glob.glob("runs/humanoid_balance/ckpts/ppo_*_steps.zip"))
    if ckpts:
        ckpt = ckpts[-1]
        vec = ckpt.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
        def make_env():
            return hb1.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=50.0)
        render_model(make_env, ckpt, vec if os.path.exists(vec) else None, "v1_final", n_steps=90)

# 3. v2 final (stable balance)
def render_v2_final():
    model_path = "runs/humanoid_balance_v2/final_model.zip"
    vec_path = "runs/humanoid_balance_v2/vecnormalize.pkl"
    if not os.path.exists(model_path):
        model_path = "runs/humanoid_balance_v2/best/best_model.zip"
        vec_path = "runs/humanoid_balance_v2/best/best_model_vecnormalize.pkl"
    # v2 env uses hb1 (367 obs)
    def make_env():
        return hb1.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=80.0)
    render_model(make_env, model_path, vec_path, "v2_final", n_steps=100, push_at=35, push_mag=80.0)

# 4. v3 final (best model - origin locked & robust)
def render_v3_final():
    model_path = "runs/humanoid_balance_v3/final_model.zip"
    vec_path = "runs/humanoid_balance_v3/vecnormalize.pkl"
    if not os.path.exists(model_path):
        model_path = "runs/humanoid_balance_v3/best/best_model.zip"
        vec_path = "runs/humanoid_balance_v3/best/best_model_vecnormalize.pkl"
    def make_env():
        return hb3.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=150.0)
    render_model(make_env, model_path, vec_path, "v3_best_final", n_steps=100, push_at=35, push_mag=150.0)

# 5. v4 final (experimental - lower reward / over-constrained posture)
def render_v4_final():
    model_path = "runs/humanoid_balance_v4/best/best_model.zip"
    vec_path = "runs/humanoid_balance_v4/best/best_model_vecnormalize.pkl"
    if not os.path.exists(model_path):
        ckpts = sorted(glob.glob("runs/humanoid_balance_v4/ckpts/ppo_*_steps.zip"))
        if ckpts:
            model_path = ckpts[-1]
            vec_path = model_path.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl")
    def make_env():
        return hb4.HumanoidBalanceEnv(render_mode="rgb_array", push_magnitude=100.0)
    render_model(make_env, model_path, vec_path, "v4_experimental", n_steps=100, push_at=35, push_mag=100.0)

if __name__ == "__main__":
    render_free_fall()
    render_v1_final()
    render_v2_final()
    render_v3_final()
    render_v4_final()
    print("Done rendering all version images!")
