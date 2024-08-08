import math

import numpy as np
import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        **kwargs,
    ):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
        #self.bn = nn.BatchNorm1d(self.batch_input)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
        hy = self.x2h(input) + self.h2h(hx)
        #hy = self.bn(hy)
        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)
        return hy


class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        bias=True,
        activation="tanh",
        device="cpu",
        **kwargs,
    ):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.activation = activation
        self.device = device

        self.rnn_cell_list = nn.ModuleList()
        assert self.activation in ["tanh", "relu"]

        self.rnn_cell_list.append(
            RNNCell(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                bias=self.bias,
                nonlinearity=self.activation,
            )
        )
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(
                RNNCell(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    bias=self.bias,
                    nonlinearity=self.activation,
                )
            )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hx=None):
        if hx is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                self.device
            )
        else:
            h0 = hx
        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](x[:, t, :], hidden[layer]) 
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1], hidden[layer]
                    )
                hidden[layer] = hidden_l
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        out = outs[-1].squeeze()
        out = self.fc(out)

        return out


class RNNClassification(nn.Module):
    def __init__(
        self,
        num_class: int = 1,
        num_frames: int = 10,
        input_length: int = 16000*6,
        hidden_size: int = 1000,
        num_layers: int = 2,
        bias: bool = True,
        activation: str = "relu",
        dropoput_rate: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ):
        super(RNNClassification, self).__init__()
        if input_length % num_frames != 0:
            raise ValueError("input_length must be divisible by num_frames.")

        self.num_frames = num_frames
        self.num_feats = input_length // num_frames
        self.batch_input = input_length // 20
        input_size = self.num_feats
        output_size = self.num_feats // 2
        self.mergeconv = nn.Conv1d(in_channels= 2 , out_channels=1 , kernel_size= 3, padding=1 ,stride= 1)
        self.rnn = SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            bias=bias,
            activation=activation,
            device=device,
        )
        self.dropout = nn.Dropout(dropoput_rate)
        self.bn = nn.BatchNorm1d(self.batch_input)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(output_size, num_class)

    def forward(self, x):
        if x.size(1) == 2:
            x = self.mergeconv(x)
        x = x.squeeze(1)
        x = x.reshape(x.size(0), self.num_frames, self.num_feats)
        out = self.rnn(x)
        #out = self.dropout(out)
        out = self.bn(out)
        out = self.relu(out)
        logit = self.linear(out)
        return logit


WaveRNN = RNNClassification

if __name__ == "__main__":
    model = WaveRNN(
        num_frames=10,
        input_length=16000*6,
    )
    x = torch.Tensor(np.random.rand(8,2, 16000*6))
    y = model(x)
    print(y.shape)
    print(y)
