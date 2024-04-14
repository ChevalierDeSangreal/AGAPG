from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import torch

def main_test():
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/paper_data/main_test_succ__thres0.5') 
    ea.Reload()
    
    batch_size = 8
    hor_dises = []
    ver_dises = []
    tot_losses = []
    dir_losses = []
    speed_losses = []
    ori_losses = []
    speeds = []
    
    for i in range(batch_size):
        hor_dis = [j.value for j in ea.scalars.Items(f'Horizon Distance{i}')]
        ver_dis = [j.value for j in ea.scalars.Items(f'Vertical Distance{i}')]
        tot_loss = [j.value for j in ea.scalars.Items(f'Total Loss{i}')]
        dir_loss = [j.value for j in ea.scalars.Items(f'Direction Loss{i}')]
        speed_loss = [j.value for j in ea.scalars.Items(f'Speed Loss{i}')]
        ori_loss = [j.value for j in ea.scalars.Items(f'Orientation Loss{i}')]
        speed = [j.value for j in ea.scalars.Items(f'Speed{i}')]
        x = [j.step for j in ea.scalars.Items(f'Horizon Distance{i}')]
        
        hor_dises.append(hor_dis)
        ver_dises.append(ver_dis)
        tot_losses.append(tot_loss)
        dir_losses.append(dir_loss)
        speed_losses.append(speed_loss)
        ori_losses.append(ori_loss)
        speeds.append(speed)
        
    
        
        
        
        
    
        
        
        
    

def main_train():
    #加载日志数据
    ea = event_accumulator.EventAccumulator('/home/cgv841/wzm/FYP/AGAPG/saved_runs/tmp_track_groundVer7__exp7__chart__42__2024-04-09 20:51:11 CST') 
    ea.Reload()
    # print(ea.scalars.Keys())

    loss_total = ea.scalars.Items('Loss')
    # loss_direction = ea.scalars.Items('Loss Direction')
    # loss_orientation = ea.scalars.Items("Loss Orientation")
    # loss_height = ea.scalars.Items("Loss Height")
    num_reset = ea.scalars.Items("Number Reset")


    x = [i.step for i in loss_total]
    y1 = [i.value for i in loss_total]
    y2 = [i.value for i in num_reset]

    fig, ax1 = plt.subplots(1, 1, figsize=(16,9), dpi=80)
    ax1.plot(x, y1, color='tab:red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue')

    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax1.grid(alpha=.4)

    ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # ax2.set_xticks(np.arange(0, len(x), 60))
    # ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
    ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
    fig.tight_layout()
    plt.savefig('tmp_plot.png')
    plt.show()

if __name__ == "__main__":
    # main_train()
    main_test()