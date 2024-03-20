import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
from aerial_gym.models import Resnet
import time
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='/home/cgv841/wzm/data', train=True,
                                        download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root='/home/cgv841/wzm/data', train=False,
                                       download=True, transform=transform)

train_size = len(train_set)
print(train_size)
test_size = len(test_set)
print(test_size)

epochs = 100
batch_size = 128
learning_rate = 2.6e-4
run_name = 'pretrain'
model = Resnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
save_path = '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/resnet.pth'

writer = SummaryWriter(f"/home/cgv841/wzm/FYP/AGAPG/runs/{run_name}")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Build data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
data_loaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": train_size, "test": test_size}

def eval_on_test_set(model):
    model.eval()
    model.to(device)
    running_accuracy = 0
    loss = 0

    for data in test_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            # print(outputs, labels, loss)
            _, preds = torch.max(outputs, 1)
            running_accuracy += torch.sum(preds == labels.data)
            
    total_loss = loss / test_size
    total_accuracy = running_accuracy / test_size
    print('Evaluation  on test set: loss{:.3f} \t accuracy = {:.2f}%'.format(total_loss, total_accuracy * 100))
    model.train()
    return total_loss, total_accuracy

def train_for_one_epoch(model):
    model.train()
    model.to(device)

    running_loss = 0
    running_accuracy = 0

    for data in train_loader:
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape, labels.shape)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        # print(loss.shape)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_accuracy += torch.sum(preds == labels.data)

    # Compute stats for the full training set
    total_loss = running_loss / train_size
    total_accuracy = running_accuracy / train_size

    return total_loss, total_accuracy

for epoch in range(epochs):
    start=time.time()
    train_loss_epoch, train_acc_epoch = train_for_one_epoch(model)
    elapsed = (time.time()-start) / 60
    print('Training epoch= {} \t cost_time= {:.2f} min \t loss= {:.3f} \t accuracy= {:.2f}%'.format(epoch, elapsed, train_loss_epoch, train_acc_epoch * 100))
    test_loss_epoch, test_acc_epoch = eval_on_test_set(model)
    writer.add_scalar('Train Loss', train_loss_epoch, epoch)
    writer.add_scalar('Train Acc', train_acc_epoch, epoch)
    writer.add_scalar('Test Loss', test_loss_epoch, epoch)
    writer.add_scalar('Test Acc', test_acc_epoch, epoch)
    
    if (epoch + 1) % 10 == 0:
        # save your trained model for the following question
        torch.save(model.state_dict(), save_path)

writer.close()
print("Training Complete!")