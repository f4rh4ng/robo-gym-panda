import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1'  # or other xxx.xxx.xxx.xxx

# initialize environment
#env = gym.make('EmptyEnvironmentPandaSim-v0', ip=target_machine_ip, gui=True)
#env = gym.make('EmptyEnvironmentURSim-v0', ip=target_machine_ip, gui=False)
env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
#env = gym.make('EndEffectorPositioningURSim-v0', ip=target_machine_ip, gui=True)
# add wrapper for automatic exception handling
env = ExceptionHandling(env)

# choose and run appropriate algorithm provided by stable-baselines
model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1500)

# save model
model.save('td3_mir_basic')

