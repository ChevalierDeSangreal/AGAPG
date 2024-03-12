import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_ground", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},
        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},


        {"name": "--num_run", "type":int, "default": 10,
            "help": "the learning rate of the optimizer"},

        {"name": "--batch_size", "type":int, "default": 1,
            "help": "will be set as equal to num_envs"},

        {"name": "--num_step", "type":int, "default": 8,
            "help": "will be set as equal to num_envs"},

        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

if __name__ == "__main__":
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.batch_size = args.num_envs

    device = args.sim_device
    print("using device:", device)
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)


    for i in range(args.num_run):
        envs.reset()
        print(f"The {i} Run Begin---------------------------")
        
        for step in range(args.num_step):
            actions = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
            new_state_ag, new_pos_tar = envs.step(actions)
            print(f"In step {step}, the position of agent is {new_state_ag[0][:9]}, the position of target is {new_pos_tar[0][:9]}")
        print(f"The {i} Run End-----------------------------")

        

    