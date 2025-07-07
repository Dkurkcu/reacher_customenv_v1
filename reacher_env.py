import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class ReacherEnv(gym.Env):
    def __init__(self, model_path="C:/Users/90546/Desktop/customenv/safe_reacher.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None

        # Action space: 2 motors
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space
        obs_high = np.inf * np.ones(self._get_obs().shape, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.max_steps_per_episode = 200
        self.current_step = 0

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        fingertip_pos = self.data.body("fingertip").xpos
        target_pos = self.data.body("target").xpos
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.data.qpos[:] = np.random.uniform(-0.1, 0.1, size=self.model.nq)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        fingertip_pos = self.data.body("fingertip").xpos
        target_pos = self.data.body("target").xpos
        self.last_dist = np.linalg.norm(fingertip_pos - target_pos)
        self.last_action = np.zeros(self.action_space.shape)

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        fingertip_pos = self.data.body("fingertip").xpos
        target_pos = self.data.body("target").xpos
        dist = np.linalg.norm(fingertip_pos - target_pos)

        reward = -dist
        if dist < 0.2:
            reward += 5
        if dist < 0.1:
            reward += 200
            terminated = True
        else:
            terminated = False

        reward += 15 * (self.last_dist - dist)
        self.last_dist = dist
        reward -= 0.05 * np.sum(np.square(action))
        reward -= 0.1 * np.sum(np.square(action - self.last_action))
        self.last_action = action
        reward -= 0.01 * np.sum(np.square(self.data.qvel))

        self.current_step += 1
        truncated = self.current_step >= self.max_steps_per_episode

        info = {
            "distance": dist,
            "TimeLimit.truncated": truncated,
            "Goal.terminated": terminated
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
