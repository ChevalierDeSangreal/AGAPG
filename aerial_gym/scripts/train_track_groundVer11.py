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
import pytz
from datetime import datetime

from torch.optim import lr_scheduler

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry
from aerial_gym.dataset import QuadGroundDataset
from aerial_gym.models import TrackGroundModelVer4
from aerial_gym.envs import LearntDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_groundVer4", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "exp5_1", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 81, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 2.6e-6,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 8,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 5000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 100,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": True, "help": "Set false to officially save the trainning log"},
        {"name": "--gamma", "type":int, "default": 0.5,
            "help": "how much will learning rate decrease"},
        {"name": "--step_size", "type":int, "default": 500,
            "help": "learning rate will decrease every step_size steps"},

        # model setting
        {"name": "--param_path_dynamic", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/dynamic_learntVer2.pth',
            "help": "The path to dynamic model parameters"},
        {"name": "--param_save_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer13.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer9__len_sample_50.pth',
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

def get_time():

    timestamp = time.time()  # 替换为您的时间戳

    # 将时间戳转换为datetime对象
    dt_object_utc = datetime.utcfromtimestamp(timestamp)

    # 指定目标时区（例如"Asia/Shanghai"）
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)

    # 将datetime对象格式化为字符串
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")

    return formatted_time_local

if __name__ == "__main__":
    args = get_args()
    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"
    # print(args.tmp)
    
    if args.tmp:
        run_name = 'tmp_' + run_name
    writer = SummaryWriter(f"/home/cgv841/wzm/FYP/AGAPG/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    device = args.sim_device
    print("using device:", device)
    envs, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dynamic = LearntDynamics(device=device, param_file_path=args.param_path_dynamic)
    dynamic.load_parameters()
    dynamic.to(device)
    
    model = TrackGroundModelVer4(device=device).to(device)
    # checkpoint = torch.load(args.param_load_path_track_simple, map_location=device)
    # model.load_state_dict(checkpoint)
    # model = TrackSimplerModel(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch} begin...")
        optimizer.zero_grad()
        
        now_quad_state = envs.reset()
        

        for step in range(args.len_sample):
        # if 1:
            # print("Step: ", step)
            
            # print(now_state, tar_pos)
            # action = model(now_state, tar_pos)]
            image = envs.get_camera_output()
            
            action = model(now_quad_state[:, 3:], image)
            
            # action = torch_rand_float(-1.0, 1.0, (args.batch_size, 4), device)
            # print(action, now_state)
            new_state_dyn = dynamic(now_quad_state, action, envs.cfg.sim.dt)
            new_state_sim, tar_state = envs.step(action)
            tar_pos = tar_state[:, :3]
            # print(new_state_dyn, tar_pos)
            

            if (step + 1) % 50 == 0:
                # print("Here I am!!")
                scaled_now_quad_pos = torch.max(new_state_dyn, torch.tensor(-10, device=device))
                scaled_now_quad_pos = torch.min(scaled_now_quad_pos, torch.tensor(10, device=device))
                loss1 = criterion(scaled_now_quad_pos[:, :2], tar_pos[:, :2])
                loss2 = torch.sum(torch.abs(scaled_now_quad_pos[:, 2] - 5)) / args.batch_size
                loss = 0.05 * loss1 + loss2
                loss.backward()
                max_norm = 1.0  # 设置梯度裁剪的阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
                new_state_dyn = new_state_dyn.detach()
                
            
            if (epoch + 1) % 10 == 0:
                if (step + 1) % 10 == 0:
            # #     # print(action)
            # #     # print(now_state)
                    print(f"    Step {step}: loss = {loss}, tar_pos = {tar_pos[0]}, now_pos = {now_quad_state[0, :3]}, new_pos = {new_state_dyn[0, :3]}, action = {action[0]}")
            
            now_quad_state = new_state_dyn
            
            if step and not (step % 50):
                envs.reset_to(now_quad_state)
            
        # dis_sim_dyn = torch.norm(new_state_dyn[:, :2] - new_state_sim[:, :2], p=2, dim=1)    
        # print("Distance between sim and dynamics:", dis_sim_dyn)
        # exit(0)
        
        scaled_now_quad_pos = torch.max(new_state_dyn, torch.tensor(-10, device=device))
        scaled_now_quad_pos = torch.min(scaled_now_quad_pos, torch.tensor(10, device=device))
        loss1 = criterion(scaled_now_quad_pos[:, :2], tar_pos[:, :2])
        loss2 = torch.sum(torch.abs(scaled_now_quad_pos[:, 2] - 5)) / args.batch_size
        loss = 0.05 * loss1 + loss2
        dis_hoz = torch.sum(torch.norm(scaled_now_quad_pos[:, :2] - tar_pos[:, :2], dim=1, p=2)) / args.batch_size
        print(f"Epoch {epoch}: loss = {loss}, ver dis = {loss2}, hor dis = {dis_hoz}")
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Vertical Distance', loss2.item(), epoch)
        writer.add_scalar('Horizen Distance', dis_hoz.item(), epoch)
        # loss.backward()
        # max_norm = 1.0  # 设置梯度裁剪的阈值
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()
        scheduler.step()
        # break
            
        # if (epoch + 1) % 100 == 0:
        #     print("Saving Model...")
            # torch.save(model.state_dict(), args.param_save_path_track_simple)
            # break
            # dynamic.save_parameters()
        # break
    
    writer.close()
    print("Training Complete!")
            

        