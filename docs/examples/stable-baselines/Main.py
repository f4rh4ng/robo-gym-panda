import sys
sys.path.append('/home/robogym/robo-gym/')
import os
import matplotlib.pyplot as plt
import numpy as np

import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling

from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

#gym.logger.set_level(40)

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1'

# Train the agent
time_steps = 2000000
best_mean_reward, n_steps = -np.inf, 0


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# initialize environment
#env = gym.make('EmptyEnvironmentPandaSim-v0', ip=target_machine_ip, gui=True)
env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)
env = Monitor(env, log_dir, allow_early_resets=True)

# Add some action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
# choose and run appropriate algorithm provided by stable-baselines
model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./TD3_panda_tensorboard/", learning_rate= 0.0003)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# train agent
model.learn(total_timesteps=int(time_steps), callback=callback, tb_log_name="second_run")
plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "End Effector Positioning panda TD3")
# plt.show()
plt.savefig('End_Effector_Positioning_panda_TD3.png')
env.kill_sim()
