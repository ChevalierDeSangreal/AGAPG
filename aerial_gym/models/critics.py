import torch
import torch.nn as nn
import torch
import torchvision.models as models


class Critics(nn.Module):
    """
    
    """
    def __init__(self, input_size=16, hidden_size1=256, hidden_size2=256, output_size=1, device='cpu'):
        print("Critics Initializing...")

        super(Critics, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size2, output_size).to(device)

        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, tar_pos):

        x = torch.cat((now_state, tar_pos), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x