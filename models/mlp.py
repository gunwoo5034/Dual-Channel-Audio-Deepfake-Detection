import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 80)
        self.bn2 = nn.BatchNorm1d(80)
        self.fc3 = nn.Linear(80, out_dim)

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1)
        y = F.relu(self.fc1(x))
        y = F.sigmoid(self.bn2(self.fc2(y)))
        y = self.fc3(y)
        return y


# class MLP(nn.Module):
#     def __init__(self, in_dim, out_dim=1, **kwargs):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256,128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 64)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.fc4 = nn.Linear(64, 32)
#         self.fc5 = nn.Linear(32, out_dim)
        
#     def forward(self,x):
#         B = x.size(0)
#         x = x.reshape(B,-1)
#         y = F.relu(self.bn1(self.fc1(x)))
#         y = F.relu(self.bn2(self.fc2(y)))
#         y = F.relu(self.bn3(self.fc3(y)))
#         y = F.sigmoid(self.fc4(y))
#         y = self.fc5(y)
        
#         return y
if __name__ == "__main__":
    model = MLP(in_dim=40 * 962, out_dim=1)
    x = torch.Tensor(np.random.rand(8, 40, 481))
    z = torch.Tensor(np.random.rand(8, 40 ,481))
    k = torch.cat((x,z),dim=2)
    y = model(k)
    print(y.shape)
    print(y)
