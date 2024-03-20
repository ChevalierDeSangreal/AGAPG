import torch
import torch.nn as nn

class TrackSimpleModel(nn.Module):
    """
    Model for Exp1
    """
    def __init__(self, input_size=15, hidden_size1=64, hidden_size2=64, hidden_size3=64, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")
        super(TrackSimpleModel, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size3, output_size).to(device)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, tar_pos):
        x = torch.cat((now_state, tar_pos), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x

class TrackSimpleModelVer2(nn.Module):
    """
    Model for Exp1
    """
    def __init__(self, input_size=15, hidden_size1=128, hidden_size2=128, hidden_size3=128, hidden_size4=128, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")
        super(TrackSimpleModelVer2, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, tar_pos):
        x = torch.cat((now_state, tar_pos), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x

class TrackSimplerModel(nn.Module):
    """
    Model for Exp1
    """
    def __init__(self, input_size=12, hidden_size1=64, hidden_size2=64, hidden_size3=128, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")
        super(TrackSimplerModel, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size3, output_size).to(device)
        self.tanh = nn.Tanh().to(device)

    def forward(self, now_state):
        x = now_state
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.output_layer(x)
        x = self.tanh(x)
        return x
    
    
class TrackSimpleModelVer3(nn.Module):
    """
    Model for Exp1
    """
    def __init__(self, input_size=12, hidden_size1=128, hidden_size2=128, hidden_size3=128, hidden_size4=128, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")
        super(TrackSimpleModelVer3, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, tar_pos):
        x = torch.cat((now_state, tar_pos), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x