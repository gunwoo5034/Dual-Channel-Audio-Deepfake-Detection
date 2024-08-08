import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x

class network_29layers(nn.Module):
    def __init__(self, in_channels , block = resblock, layers=[1, 2, 3, 4] , num_classes=1):
        super(network_29layers, self).__init__()
        self.conv1_d  = mfm(in_channels= in_channels, out_channels= 48, kernel_size=5, stride= 1, padding= 2)
        self.conv1_r  = mfm(in_channels= in_channels, out_channels= 48, kernel_size=5, stride= 1, padding= 2)
        self.conv1 = mfm(in_channels= in_channels, out_channels= 48, kernel_size=5, stride= 1, padding= 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(in_channels=48, out_channels=96, kernel_size=3, stride=1,padding= 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(in_channels=96, out_channels=192,kernel_size= 3, stride=1,padding= 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(in_channels=192, out_channels=128,kernel_size= 3,stride= 1, padding= 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(in_channels=128, out_channels = 128,kernel_size= 3,stride= 1, padding= 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # self.fc     = mfm(in_channels=11904, out_channels= 256, type=0)
        self.fc     = mfm(in_channels=23424, out_channels= 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)
        self.mergeconv = nn.Conv2d(in_channels= 2 , out_channels=1 , kernel_size= 3, padding=1 ,stride= 1)
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x_d = x[:, 0, : , :]
        # x_d = x_d.unsqueeze(1)
        # x_r = x[:, 1, : , :]
        # x_r = x_r.unsqueeze(1)
        # x_d = self.conv1_d(x_d)
        # x_r = self.conv1_r(x_r)
        if x.size(1) == 2:
            x = self.mergeconv(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out




def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model

def LightCNN_29Layers(**kwargs):
    model = network_29layers(1, resblock, [1, 2, 3, 4], **kwargs)
    return model

if __name__ == "__main__":
    model = LightCNN_29Layers()
    x = torch.Tensor(np.random.rand(8, 40, 481))
    y = torch.Tensor(np.random.rand(8, 40, 481))
    z = torch.cat((x,y),dim=2)
    out = model(z)
    print(out.shape)
