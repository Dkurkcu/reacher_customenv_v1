from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from reacher_env import ReacherEnv

# Wrap the env
env = DummyVecEnv([lambda: ReacherEnv("C:/Users/90546/Desktop/customenv/custom_env.xml")])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=100_000)

# Save
model.save("ppo_reacher")
print("model saved")


