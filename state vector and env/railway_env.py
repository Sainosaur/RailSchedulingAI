import gymnasium as gym
import numpy as np


class RailwayEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(  # set upper and lower limit
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([5, 5, 2, 2, 500, 20, 20], dtype=np.float32),
        )
        self.action_space = gym.spaces.Discrete(
            4
        )  # generate 0/1/2/3 with particular probability(should be adjusted later)
        self.reset()

    def reset(self, seed=None, options=None):  # reset state vector to default setup
        # pos = 0: terminus A, 1: loop 1, 2: loop 2, 3: loop 3, 4: loop 4, 5: terminus B
        self.train1_pos = 0  # start at terminus A
        self.train2_pos = 5  # start at terminus B
        # speed = 0: stop, 1: slow, 2: full
        self.train1_speed = 1
        self.train2_speed = 1
        # delay 0-50: latency against timetable in second
        self.train1_delay = 0
        self.train2_delay = 0
        self.step_count = 0  #
        return self._get_obs(), {}

    def _get_obs(self):
        headway = (
            abs(self.train1_pos - self.train2_pos) * 10
        )  # simple headway calculation
        return np.array(
            [
                self.train1_pos,
                self.train2_pos,
                self.train1_speed,
                self.train2_speed,
                headway,
                self.train1_delay,
                self.train2_delay,
            ],
            dtype=np.float32,
        )

    def step(self, action):  # determine action for values correspond to action space
        reward = 0  # reward function required
        terminated = False
        if action == 0:
            self.train1_speed = 0  # hold train 1
        elif action == 1:
            self.train1_speed = 2  # proceed train 1 with full speed
        elif action == 2:
            self.train2_speed = 0  # hold train 2
        elif action == 3:
            self.train2_speed = 2  # proceed train 2 with full speed

        # Move trains (can be optimised later)
        self.train1_pos += self.train1_speed
        self.train2_pos -= self.train2_speed

        # Clamp to track boundaries
        self.train1_pos = min(self.train1_pos, 5)
        self.train2_pos = max(self.train2_pos, 0)

        self.step_count += 1
        truncated = self.step_count >= 500  # ending an episode after too many steps
        if (
            self.train1_pos == 5 and self.train2_pos == 0
        ):  # stop if two trains reach terminus
            terminated = True
        return self._get_obs(), reward, terminated, truncated, {}


# test script (train 1 reaches B and train 2 reaches A)
env = RailwayEnv()
obs, _ = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Action: {action}")
    if done:
        break
