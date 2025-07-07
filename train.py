import os
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from reacher_env import ReacherEnv

# ✅ Initialize wandb
wandb.init(
    project="Reacher_project_deniz",
    name="ppo_reacher_run",
    config={
        "algo": "PPO",
        "timesteps": 200_000,
        "learning_rate": 1e-4,
        "reward_shaping": "precision_bonus + progress + penalties",
        "env": "Custom Reacher v1"
    }
)

# ✅ Custom wandb callback
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals["rewards"][0]
        wandb.log({
            "distance_to_target": info.get("distance"),
            "reward": reward
        }, step=self.num_timesteps)
        return True

# ✅ Create environment (no video wrapper)
env = DummyVecEnv([
    lambda: Monitor(ReacherEnv("C:/Users/90546/Desktop/customenv/safe_reacher.xml"))
])

# ✅ PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)

# ✅ Train
model.learn(
    total_timesteps=200_000,
    callback=CustomWandbCallback()
)

# ✅ Save model
os.makedirs("./models", exist_ok=True)
model_path = "./models/ppo_reacher_final"
model.save(model_path)

# ✅ Log model
artifact = wandb.Artifact("ppo_reacher_final", type="model")
artifact.add_file(f"{model_path}.zip")
wandb.log_artifact(artifact)

print("Training complete. Model saved and logged to wandb.")
