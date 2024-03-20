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

sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
# print(sys.path)
from aerial_gym.envs import *
from aerial_gym.utils import task_registry
from aerial_gym.dataset import QuadGroundDataset
from aerial_gym.models import TrackGroundModelVer3
from aerial_gym.envs import LearntDynamics
# os.path.basename(__file__).rstrip(".py")
def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "track_ground", "help": "The name of the task."},
        {"name": "--experiment_name", "type": str, "default": "exp3_4", "help": "Name of the experiment to run or load."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--num_envs", "type": int, "default": 8, "help": "Number of environments to create. Batch size will be equal to this"},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed. Overrides config file if provided."},

        # train setting
        {"name": "--learning_rate", "type":float, "default": 0.00026,
            "help": "the learning rate of the optimizer"},
        {"name": "--batch_size", "type":int, "default": 8,
            "help": "batch size of training. Notice that batch_size should be equal to num_envs"},
        {"name": "--num_worker", "type":int, "default": 4,
            "help": "num worker of dataloader"},
        {"name": "--num_epoch", "type":int, "default": 4000,
            "help": "num of epoch"},
        {"name": "--len_sample", "type":int, "default": 50,
            "help": "length of a sample"},
        {"name": "--tmp", "type": bool, "default": True, "help": "Set false to officially save the trainning log"},

        # model setting
        {"name": "--param_path_dynamic", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/dynamic_learntVer2.pth',
            "help": "The path to dynamic model parameters"},
        {"name": "--param_save_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer4.pth',
            "help": "The path to model parameters"},
        {"name": "--param_load_path_track_simple", "type":str, "default": '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/track_groundVer3_saved.pth',
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
    model = TrackGroundModelVer3(device=device).to(device)
    # checkpoint = torch.load(args.param_load_path_track_simple)
    # model.load_state_dict(checkpoint)
    # model = TrackSimplerModel(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    criterion = nn.MSELoss()

    # tar_pos = torch.tensor([0, 0, 3], dtype=torch.float32).to(device)
    # # 使用 unsqueeze 在第一维度上添加一个维度
    # tar_pos = torch.unsqueeze(tar_pos, 0).expand(args.batch_size, -1)

    # tar_ang = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
    # tar_ang = torch.unsqueeze(tar_ang, 0).expand(args.batch_size, -1)

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
            
            tmp_loss = criterion(new_state_dyn[:, :2], tar_pos[:, :2])
            if (epoch + 1) % 10 == 0:
                if (step + 1) % 10 == 0:
            # #     # print(action)
            # #     # print(now_state)
                    print(f"    Step {step}: loss = {tmp_loss}, tar_pos = {tar_pos[0]}, now_pos = {now_quad_state[0, :3]}, new_pos = {new_state_dyn[0, :3]}, action = {action[0]}")
                
                # writer.add_scalar('Distance', dis.item(), epoch * args.len_sample + step)
            # optimizer.step()
            # new_state_sim = envs.step(action)
            # dis = torch.norm(new_state_dyn - new_state_sim)

            now_quad_state = new_state_dyn
        loss1 = criterion(new_state_dyn[:, :2], tar_pos[:, :2])
        loss2 = torch.sqrt(torch.sum((new_state_dyn[:, 3] - 5) ** 2)) / args.batch_size
        loss =  loss1
        print(f"Epoch {epoch}: loss = {loss}, loss2 = {loss2}")
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Loss_height_distance', loss2.item(), epoch)
        loss.backward()
        # dis.backward()
        max_norm = 1.0  # 设置梯度裁剪的阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # break
            
        if (epoch + 1) % 100 == 0:
            print("Saving Model...")
            torch.save(model.state_dict(), args.param_save_path_track_simple)
            # break
            # dynamic.save_parameters()
        # break
    
    writer.close()
    print("Training Complete!")
            

        