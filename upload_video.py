import wandb

# ✅ Initialize wandb (start a run)
wandb.init(
    project="Reacher_project_deniz",  # Your wandb project name
    name="mujoco_viewer_recording"    # This run's name
)

# ✅ Log your local video file
wandb.log({
    "mujoco_run_video": wandb.Video("C:/Users/90546/Videos/2025-07-07 12-33-08.mp4")
})

print("Video uploaded to wandb!")
