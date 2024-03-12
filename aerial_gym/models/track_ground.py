import torch
import torch.nn as nn
import torch
import torchvision.models as models


class TrackGroundModel(nn.Module):
    """
    
    """
    def __init__(self, input_size=16, hidden_size1=128, hidden_size2=128, hidden_size3=128, hidden_size4=128, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")

        super(TrackGroundModel, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
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

class TrackGroundModelVer2(nn.Module):
    """
    
    """
    def __init__(self, input_size=16, hidden_size1=64, hidden_size2=64, hidden_size3=64, output_size=4, device='cpu'):
        print("SimpleTrackModel Initializing...")

        super(TrackGroundModelVer2, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size3, output_size).to(device)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
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

if __name__ == "__main__":
    # 获取 ResNet-18 模型
    resnet18_model = models.resnet18(pretrained=True)
    in_features = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(in_features, 4)

    # 输出模型结构
    print(resnet18_model)