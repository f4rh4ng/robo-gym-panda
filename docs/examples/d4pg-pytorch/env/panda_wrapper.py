import sys
sys.path.append('/home/robogym/robo-gym/')
import gym
import dmc2gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling


class PandaWrapper:
    def __init__(self, agent_type):
        self.agent_type = agent_type
        if agent_type == "exploration":
            target_machine_ip = '127.0.0.1'
            self.env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=False)
            self.env = ExceptionHandling(self.env)
        else:
            target_machine_ip = '127.0.0.1'
            self.env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
            self.env = ExceptionHandling(self.env)

    def reset(self):
        state = self.env.reset()
        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward

