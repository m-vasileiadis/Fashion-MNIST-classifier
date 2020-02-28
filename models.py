import torch
from torch import nn
from torchvision import models


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# simple cnn architecture 2 conv/bn/relu + 2 fc
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(7 * 7 * 64, 7 * 7 * 64),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7 * 7 * 64, 10))

        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# small shufflenet v2 architecture
# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
# https://arxiv.org/abs/1807.11164
class shufflenet(nn.Module):
    def __init__(self):
        super(shufflenet, self).__init__()

        # input must be resized to 56x56
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        vanilla_model = models.shufflenet_v2_x0_5(pretrained=False) # it wouldnt be "fair" to start with a pretrained model

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            vanilla_model._modules['maxpool'],
            vanilla_model._modules['stage2'],
            vanilla_model._modules['stage3'],
            vanilla_model._modules['stage4'],
            vanilla_model._modules['conv5'])

        self.fc = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(2 * 2 * 1024, 10))

        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        out = self.bilinear(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# small mixnet architecture
# MixConv: Mixed Depthwise Convolutional Kernels
# https://arxiv.org/abs/1907.09595
class mixnet(nn.Module):
    def __init__(self):
        super(mixnet, self).__init__()

        # input must be resized to 56x56
        self.bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        vanilla_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'mixnet_s', pretrained=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            vanilla_model._modules['blocks'],
            vanilla_model._modules['conv_head'],
            vanilla_model._modules['bn2'],
            vanilla_model._modules['act2'],
            vanilla_model._modules['global_pool'])

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1536, 10))

        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)


    def forward(self, x):
        out = self.bilinear(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out