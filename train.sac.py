import os
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from reacher_env import ReacherEnv

# ✅ Check CUDA availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠ GPU not available — using CPU")

# ✅ Initialize wandb
wandb.init(
    project="Reacher_project_deniz",
    name="sac_reacher_run1",
    config={
        "algo": "SAC",
        "timesteps": 300_000,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "net_arch": [256, 256],
        "device": device,
        "env": "Custom Reacher v1"
    }
)

# ✅ Wandb logging callback
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals["rewards"][0]
        distance = info.get("distance")
        data = {"reward": reward}
        if distance is not None:
            data["distance_to_target"] = distance
        wandb.log(data, step=self.num_timesteps)
        return True

# ✅ Create the environment
env = DummyVecEnv([lambda: Monitor(ReacherEnv("C:/Users/90546/Desktop/customenv/safe_reacher.xml"))])

# ✅ Create SAC model with GPU
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    batch_size=256,
    train_freq=1,
    gradient_steps=-1,
    policy_kwargs=dict(net_arch=[256, 256]),
    device=device,   # Use GPU if available
    verbose=1,
    
)

# ✅ Confirm model device
print(f"✅ SAC model using device: {model.policy.device}")

# ✅ Train the model
model.learn(
    total_timesteps=100_000,
    callback=CustomWandbCallback()
)

# ✅ Save the model
os.makedirs("./models", exist_ok=True)
model_path = "./models/sac_reacher"
model.save(model_path)

# ✅ Log model artifact to wandb
artifact = wandb.Artifact("sac_reacher", type="model")
artifact.add_file(f"{model_path}.zip")
wandb.log_artifact(artifact)

print("✅ SAC training complete. Model saved and logged to wandb.")
