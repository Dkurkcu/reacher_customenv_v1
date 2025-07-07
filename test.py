from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from reacher_env import ReacherEnv

# Create the environment
env = DummyVecEnv([lambda: ReacherEnv("C:/Users/90546/Desktop/customenv/safe_reacher.xml"),])

# Load the trained model
model = PPO.load("ppo_reacher", env=env)


# Reset environment

obs = env.reset()
for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env.envs[0].render()

    if done[0]:
        # Print the info dict — this will show if truncated or terminated
        print(f"Episode ended — info: {info[0]}")
        obs = env.reset()


# Clean up
env.close()
