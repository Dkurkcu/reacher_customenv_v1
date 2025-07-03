from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from reacher_env import ReacherEnv

# Create the environment
env = DummyVecEnv([lambda: ReacherEnv("C:/Users/90546/Desktop/customenv/custom_env.xml")])

# Load the trained model
model = PPO.load("ppo_reacher", env=env)


# Reset environment
obs = env.reset()

# Run for 1000 steps
for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Render the simulation
    env.envs[0].render()  # Because we're using DummyVecEnv
    
    if done[0]:
        obs = env.reset()

# Clean up
env.close()
