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