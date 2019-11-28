import torch
import torch.nn as nn
from torch.nn import functional as F


class Toynetwork(nn.Module):
    def __init__(self):
        super(Toynetwork, self).__init__()
        self.input_layer = nn.Conv2d(3, 3, kernel_size=2)
        self.output = nn.Linear(2883, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.output(x))
        return x