from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from reacher_env import ReacherEnv

# === Choose algorithm and model file ===
# Uncomment one of these as needed

# For PPO:
ALGO = PPO
MODEL_PATH = "C:/Users/90546/Desktop/customenv/models/ppo_reacher.zip"

# For SAC:
#ALGO = SAC
#MODEL_PATH = "C:/Users/90546/Desktop/customenv/models/sac_reacher.zip"

# === Create environment ===
env = DummyVecEnv([lambda: ReacherEnv("C:/Users/90546/Desktop/customenv/safe_reacher.xml")])

# === Load model ===
model = ALGO.load(MODEL_PATH, env=env)

# === Run and visualize ===
obs = env.reset()

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Render MuJoCo viewer
    env.envs[0].render()

    # When an episode ends
    if done[0]:
       
        obs = env.reset()

# === Clean up ===
env.close()
