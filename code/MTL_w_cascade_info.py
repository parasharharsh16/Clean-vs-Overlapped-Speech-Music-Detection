import torch
from torch import nn
from pytorch_tcn import TCN
import numpy as np


class MtlCascadeModel(nn.Module):
    def __init__(self, hp):
        super(MtlCascadeModel, self).__init__()
        # TCN parameters
        self.num_channels = [63] * hp["n_layers"]
        self.kernel_size = 3
        self.dropout = np.random.uniform(0.05, 0.5)
        self.dilations = [2**nd for nd in self.num_channels]
        self.tcn = TCN(
            128,
            self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            # dilations=self.dilations,
        )
        self.linear = nn.Linear(
            self.num_channels[-1] * self.num_channels[-1], hp["sp_hidden_nodes"]
        )
        # self.tcn = TCN(128, self.num_channels, self.kernel_size, dilations=dilations, dropout=self.dropout)

        # Dense layers for each output
        self.dense_sp = self.make_layers(hp["sp_hidden_nodes"], hp["n_sp_hidden_lyrs"])
        self.dense_mu = self.make_layers(hp["mu_hidden_nodes"], hp["n_mu_hidden_lyrs"])
        self.dense_smr = self.make_layers(
            hp["smr_hidden_nodes"], hp["n_smr_hidden_lyrs"]
        )

        # Output layers
        self.output_sp = nn.Linear(hp["sp_hidden_nodes"], 1)
        self.output_mu = nn.Linear(hp["mu_hidden_nodes"], 1)
        self.output_smr = nn.Linear(hp["smr_hidden_nodes"], 2)

        # self.softmax = nn.Softmax(dim=1)
        # self.softmax2 = nn.Softmax(dim=1)

    def make_layers(self, hidden_nodes, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.BatchNorm1d(hidden_nodes))
            layers.append(nn.Dropout(0.4))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        # TCN
        # x = x.transpose(1, 2)
        y1 = self.tcn(x)
        y2 = y1.view(y1.size(0), -1)
        # Flatten TCN output
        y3 = self.linear(y2)
        # y1 = torch.transpose(y1, 0, 1)

        # y1 = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        # y1 = y1.contiguous().view(y1.size(0), -1)

        # Dense layers
        # print(y2.shape)
        # print(y3.shape)
        y2_sp = self.dense_sp(y3)
        y2_mu = self.dense_mu(y3)
        y2_smr = self.dense_smr(y3)

        # Output layers
        y3_sp = self.output_sp(y2_sp)
        y3_sp = torch.sigmoid(y3_sp)
        y3_mu = self.output_mu(y2_mu)
        y3_mu = torch.sigmoid(y3_mu)
        y3_smr = self.output_smr(y2_smr)
        y3_smr = torch.sigmoid(y3_smr)

        # print(y3_smr.shape)
        # print(y3_smr)
        # out_sp = self.softmax(y3_sp)
        # out_mu = self.softmax(y3_mu)
        # out_smr = self.softmax(y3_smr)
        return y3_sp, y3_mu, y3_smr


