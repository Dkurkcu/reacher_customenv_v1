import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class ReacherEnv(gym.Env):
    def __init__(self, model_path="C:/Users/90546/Desktop/customenv/custom_env.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Action space: 2 joint motors
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: joint angles, velocities, fingertip pos, target pos
        obs_high = np.inf * np.ones(self._get_obs().shape, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.viewer = None

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        fingertip_pos = self.data.body("fingertip").xpos
        target_pos = self.data.body("target").xpos
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.data.qpos[:] = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        self.data.qvel[:] = 0.0
         # Randomize target position (in X, Y plane)(optional)
        xy = np.random.uniform(low=-1.0, high=1.0, size=2)
        self.model.body("target").pos[:2] = xy
        self.model.body("target").pos[2] = 0.2  # Keep Z fixed at 0.2
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        fingertip_pos = self.data.body("fingertip").xpos
        target_pos = self.data.body("target").xpos
        dist = np.linalg.norm(fingertip_pos - target_pos)

        reward = -dist - 0.01 * np.sum(np.square(self.data.ctrl))
        terminated = dist < 0.05    # True if goal is reached
        truncated = False           # or add a time limit if you like

        info = {"distance": dist}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
