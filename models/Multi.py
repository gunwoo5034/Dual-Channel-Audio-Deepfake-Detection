import torch
import torch.nn as nn
from models.tssd import RSM1D
#from tssd import RSM1D
import torch.nn.functional as F
#from tssd import TSSD
#from mlp import MLP
class Multimodal(nn.Module):
    def __init__(self,
                 in_channels=5 * 10,
                 in_channels_wave = 1, 
                 in_dim = 16000 * 10,
                 input_shape = 40 * 801,
                 num_frames=10, 
                 input_len=16000*10, 
                 hidden_dim=30, 
                 out_dim=30,
                 image_model = "CNN",
                 audio_model = "TSSD",
                 **kwargs):
        super(Multimodal, self).__init__()
        # self.num_frames = num_frames
        # self.num_feats = input_len // num_frames
        # self.lstm=nn.LSTM(input_size=self.num_feats, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.01,)
                
        
        self.image_layer_CNN=nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2,stride=2),
                                       nn.Flatten(),
                                       nn.Linear(128*28*28,4*28*28),
                                       nn.ReLU(), # Relu 넣어봄
                                       nn.Linear(4*28*28,30),
                                       nn.ReLU() # Relu 넣어봄
                                       )
        
        # self.audio_layer_TSSD=nn.Sequential(nn.Conv1d(in_channels=in_channels_wave, out_channels=16, kernel_size=7, padding=3, bias=False),
        #                                     nn.BatchNorm1d(16),
        #                                     nn.ReLU(),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=16, channels_out=32),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=32, channels_out=64),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=64, channels_out=128),
        #                                     nn.MaxPool1d(kernel_size=4),
        #                                     RSM1D(channels_in=128,channels_out=128),
        #                                     nn.MaxPool1d(kernel_size=252),
        #                                     nn.Flatten(start_dim=1),
        #                                     nn.Linear(in_features=128,out_features=64),
        #                                     nn.ReLU(),
        #                                     nn.Linear(64,30),
        #                                     nn.ReLU())
        self.audio_layer_TSSD = TSSD(in_channels= in_channels_wave , in_dim = in_dim)
        #self.aduio_layer_MLP  = MLP(in_dim= input_shape , out_dim =1)
        
        
        self.audio_layer_MLP=nn.Sequential(nn.Linear(input_shape,120),
                                           nn.ReLU(),
                                           nn.Linear(120, 80),
                                           nn.Sigmoid(),
                                           nn.Linear(80, 30))
        
        self.multi_layer=nn.Sequential(nn.Linear(60,1))
        
        self.audio_uni_layer=nn.Sequential(nn.Linear(30,1))
        
        self.image_uni_layer=nn.Sequential(nn.Linear(30,1))
        self.audio_model = audio_model
        self.image_model = image_model

        
        
    

    def forward(self, image_input:torch.Tensor, audio_input:torch.Tensor):
        # 이미지 부분 
        if self.image_model=="CNN":
            image_output=self.image_layer_CNN(image_input) # 콘캣 위한 fc1,2 하나
        
        # 오디오 부분 
        if self.audio_model=="TSSD": #tssd -> 오디오 부분
            #audio_input=audio_input.squeeze(1)
            audio_output = self.audio_layer_TSSD(audio_input)
        
        if self.audio_model=="MLP":
            batch=audio_input.size(0)
            audio_input=audio_input.reshape(batch, -1)
            audio_output=self.audio_layer_MLP(audio_input)

        
        
        # 이미지+ 오디오 concat 부분
        multi_input=torch.cat([image_output,audio_output],dim=1)
        multi_output=self.multi_layer(multi_input)
        
        # 이미지, 오디오 각각에 해당 Out put
        image_output=self.image_uni_layer(image_output)
        audio_output=self.audio_uni_layer(audio_output)
        return image_output, audio_output, multi_output



class TSSD(nn.Module):  # Res-TSSDNet
    def __init__(self, in_dim,in_channels =1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=16, kernel_size=7, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=30)
        #self.out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1])

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.out(x)
        return x
    
    
# image_input=torch.rand(16, 50, 224, 224)
# audio_input=torch.rand(16, 1, 16000 * 10)
# model=Multimodal()
# a,b,c=model(image_input,audio_input)

#random_torch=random_torch.reshape(4,2)
#random_torch.size()
# random_torch1=torch.rand(16,3,224,224)
# random_torch2=torch.rand(16,25*3,224,224)
# torch_stack=torch.cat([random_torch,random_torch1,random_torch2],dim=1) 채널을 콘캣하는 방법.?
# torch_stack.size()