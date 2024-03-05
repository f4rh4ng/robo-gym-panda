import sys
sys.path.append('/home/robogym/robo-gym/')
import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import TD3
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

# initialize environment

env = gym.make('FollowPandaSim-v0', ip=target_machine_ip, gui=True)

# add wrapper for automatic exception handling
env = ExceptionHandling(env)


# Evaluating policy

model = TD3.load('best_model', env, tensorboard_log="./TD3_panda_eval_logs/")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Run trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

