import gymnasium as gym
from validator import ValidationLayer
from reward_function import compute_reward, TrainState

class ModernizedLine104(gym.Env):
    def __init__(self):
        super().__init__()
        self.vl = ValidationLayer()
        self.action_space = gym.spaces.Discrete(4)
        # State: [pos, speed, dtz, aspect]
        self.observation_space = gym.spaces.Box(low=0, high=80000, shape=(4,))

    def step(self, action):
        # 1. Validate with Sieve
        safe_action, overridden = self.vl.get_safe_action(action, self.x, self.v, self.dtz)

        # 2. Physics Update (0.5s)
        prev_x = self.x
        # ... (SUVAT updates for self.x and self.v based on safe_action) ...

        # 3. Reward with "Beat Up" logic
        state = TrainState(current_position=self.x, previous_position=prev_x, ...)
        reward_out = compute_reward(state)
        total_reward = reward_out.r_total

        if overridden:
            total_reward -= 500 # The Penalty

        return self._get_obs(), total_reward, self.x >= 76651, False, {"overridden": overridden}
