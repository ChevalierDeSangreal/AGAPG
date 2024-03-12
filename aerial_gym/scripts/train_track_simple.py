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
from torch.utils.data import DataLoader
import sys

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry
from aerial_gym.dataset import QuadGroundDataset
from aerial_gym.models import TrackSimpleModel, TrackSimplerModel
from aerial_gym.envs import LearntDynamics

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_ground", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "default": 128, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 0.1,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 128,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 1000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 150,
            "help": "length of a sample"},

        # model setting
        {"name": "--param_path_dynamic", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/dynamic_learntVer2.pth',
            "help": "The path to dynamic model parameters"},
        {"name": "--param_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_simpler.pth',
            "help": "The path to model parameters"},
        
        ]

    # parse arguments
    args = gymutil.parse_arguments(
        description="APG Policy",
        custom_parameters=custom_parameters)

    assert args.batch_size == args.num_envs, "batch_size should be equal to num_envs"

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"


    return args



if __name__ == "__main__":
    args = get_args()
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.sim_device
    print("using device:", device)

    envs, env_cfg = task_registry.make_env(name=args.task, args=args)


    # dataset = QuadGroundDataset(args.num_sample, args.len_sample, device)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    dynamic = LearntDynamics(device=device, param_file_path=args.param_path_dynamic)
    dynamic.load_parameters()
    dynamic.to(device)
    # model = TrackSimpleModel(device=device).to(device)
    model = TrackSimplerModel(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()

    tar_pos = torch.tensor([0, 0, 3], dtype=torch.float32).to(device)
    # 使用 unsqueeze 在第一维度上添加一个维度
    tar_pos = torch.unsqueeze(tar_pos, 0).expand(args.batch_size, -1)

    tar_ang = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
    tar_ang = torch.unsqueeze(tar_ang, 0).expand(args.batch_size, -1)

    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch} begin...")
        
        now_state = envs.reset()

        for step in range(args.len_sample):
        # if 1:
            # print("Step: ", step)
            optimizer.zero_grad()
            # print(now_state, tar_pos)
            # action = model(now_state, tar_pos)
            action = model(now_state)
            
            # action = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
            # print(action, now_state)
            new_state_dyn = dynamic(now_state, action, envs.cfg.sim.dt)
            # print(new_state_dyn, tar_pos)
            loss = criterion(new_state_dyn[:, 3:6], tar_ang)
            # loss = criterion(new_state_dyn[:, :3], tar_pos)

            # optimizer.step()
            new_state_sim = envs.step(action)
            dis = torch.norm(new_state_dyn - new_state_sim)
            loss.backward()
            # dis.backward()
            optimizer.step()
            now_state = new_state_sim
            if (epoch + 1) % 10 == 0:
                # print(action)
                # print(now_state)
                print(f"    Step {step}: loss = {loss}, distance between sim and dyn = {dis}")
                writer.add_scalar('Loss', loss.item(), epoch * args.len_sample + step)
                writer.add_scalar('Distance', dis.item(), epoch * args.len_sample + step)
            
        if (epoch + 1) % 100 == 0:
            print("Saving Model...")
            torch.save(model.state_dict(), args.param_path_track_simple)
            # dynamic.save_parameters()
        # break
    
    writer.close()
    print("Training Complete!")
            

        