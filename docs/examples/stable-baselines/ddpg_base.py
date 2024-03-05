import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import gym
# gym.logger.set_level(40)
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from datetime import datetime
# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1'

# initialize environment (to render the environment set gui=True)
#env = gym.make('EndEffectorPositioningURSim-v0', ip=target_machine_ip, gui=True)
#env = gym.make('EmptyEnvironmentPandaSim-v0', ip=target_machine_ip, gui=True)
#env = gym.make('EndEffectorPositioningPandaSim-v0', ip=target_machine_ip, gui=True)
#env = gym.make('AvoidanceRaad2022URSim-v0', ip=target_machine_ip, gui=True)
env = gym.make('FollowPandaSim-v0', ip=target_machine_ip, gui=True)
env = ExceptionHandling(env)
model = TD3(MlpPolicy, env, verbose=1)
# follow the instructions provided by stable-baselines

num_episodes = 60
start_time = datetime.now()

for episode in range(num_episodes):
    print("running the episode number", episode)
    model.learn(total_timesteps=int(15000))
    # saving and loading a model
    model.save("td3_ur3e")
    del model
    print("load the model", episode)
    model = TD3.load("td3_ur3e", env=env, policy=MlpPolicy)
    print('Duration: {}'.format(datetime.now() - start_time))
    
print('Duration: {}'.format(datetime.now() - start_time))
env.kill_sim()
