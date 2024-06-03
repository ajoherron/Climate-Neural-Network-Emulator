import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Change the first convolution layer to match the number of input channels
        resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Remove the last fully connected layer of ResNet-18
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Define a new fully connected layer to match the output dimensions of the provided CNNModel
        self.fc = nn.Linear(resnet.fc.in_features, output_channels * 90 * 144)
        self.output_channels = output_channels  # Store output_channels as an attribute

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 90, 144, 1)
        return x


class CNNModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(10, output_channels * 90 * 144)
        self.output_channels = output_channels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 90, 144, 1)
        return x
