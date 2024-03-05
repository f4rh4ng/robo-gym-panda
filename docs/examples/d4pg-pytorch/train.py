import gym
import argparse
from models.engine import load_engine
from utils.utils import read_config

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, help="Path to the config file.")
# parser.add_argument("--config", type=str, default='./configs/openai/d4pg/panda_reach_d4pg.yml', help="Path to the config file.")
# python3 train.py --config ./configs/openai/d4pg/panda_reach_d4pg.yml

if __name__ == "__main__":

    args = vars(parser.parse_args())
    config = read_config(args['config'])
    engine = load_engine(config)
    #Train
    #engine.train()
    # Train fom checkpoint
    #engine.train_from_checkpoint()
    #test
    engine.test()
