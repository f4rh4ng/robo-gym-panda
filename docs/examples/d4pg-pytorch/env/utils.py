from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .env_wrapper import EnvWrapper
from .lunar_lander_continous import LunarLanderContinous
from .panda_wrapper import PandaWrapper
from .panda_follow_wrapper import PandaFollowWrapper


def create_env_wrapper(config, agent_type):
    env_name = config['env']
    if env_name == "Pendulum-v0":
        return PendulumWrapper(config)
    elif env_name == "BipedalWalker-v2":
        return BipedalWalker(config)
    elif env_name == "LunarLanderContinuous-v2":
        return LunarLanderContinous(config)
    elif env_name == "EndEffectorPositioningPandaSim-v0":  # Farhang: adding Panda reach environment
        return PandaWrapper(agent_type)
    elif env_name == "FollowPandaSim-v0":  # Farhang: adding Panda follow environment
        return PandaFollowWrapper(agent_type)
    return EnvWrapper(env_name)

