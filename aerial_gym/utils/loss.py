import torch.nn as nn
import torch

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


def pav_loss(quad_state, tar_pos, criterion):
    vel_hoz = quad_state[:, 6:9]
    acc_hoz = quad_state[:, 9:12]
    
    dis_hoz = (tar_pos[:, :2] - quad_state[:, :2])
    
    
    loss_vel_hoz = criterion(vel_hoz, dis_hoz)
    loss_acc_hoz = criterion(acc_hoz, dis_hoz)
    loss_dis_hoz = criterion(tar_pos[:, :2], quad_state[:, :2])
    
    loss = 0.8 * loss_dis_hoz + 0.15 * loss_vel_hoz + 0.05 * loss_acc_hoz
    return loss