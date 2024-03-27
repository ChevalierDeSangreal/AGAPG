import torch.nn as nn
import torch
from torch import tensor

def acc_loss(quad_state, tar_pos, criterion):
    acc = quad_state[:, 6:8]
    # print(tar_pos.shape, quad_state.shape)
    dis = (tar_pos[:, :2] - quad_state[:, :2])
    return criterion(acc, dis)

def acch_loss(quad_state, tar_pos, tar_height, criterion):
    acc = quad_state[:, 6:9]
    dis_hoz = (tar_pos[:, :2] - quad_state[:, :2])
    dis_ver = torch.tensor(tar_height) - quad_state[:, 2]
    dis_ver = torch.unsqueeze(dis_ver, dim=1)
    # print(dis_hoz.shape, dis_ver.shape)
    dis = torch.cat((dis_hoz, dis_ver), dim=1)
    return criterion(acc, dis)


def pav_loss(quad_state, tar_pos, tar_h, criterion):
    vel = quad_state[:, 6:9]
    acc = quad_state[:, 9:12]

    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    
    tar_pos = tar_pos[:, :2]
    tar_pos = torch.cat((tar_pos, z_coords), dim=1)
    # print(tar_pos.shape, quad_state[:, :3].shape)
    dis = (tar_pos - quad_state[:, :3])
    
    loss_vel = criterion(vel, dis)
    loss_acc = criterion(acc, dis)
    loss_dis = criterion(tar_pos, quad_state[:, :3])
    
    loss = 0.8 * loss_dis + 0.15 * loss_vel + 0.05 * loss_acc
    return loss

def velh_loss(quad_state, tar_pos, tar_h):
    vel = quad_state[:, 6:9]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    dis = (tar_pos - quad_state[:, :3])
    
    loss = torch.norm(vel - dis, dim=1, p=2)
    return loss

def velh_lossVer2(quad_state, tar_pos, tar_h, criterion):
    vel = quad_state[:, 6:9]
    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_pos[:, :2], z_coords), dim=1)
    dis = (tar_pos - quad_state[:, :3])
    # dis[:, 2] *= 0.1
    
    return criterion(dis, vel)