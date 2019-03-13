
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import time


HIDDEN = 50

class Net(nn.Module):
    def __init__(self, net_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 30, kernel_size=12)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(30, HIDDEN)
        self.fc1normal = nn.Linear(30, HIDDEN)
        self.fc1negative = nn.Linear(30, HIDDEN)

        self.fc2 = nn.Linear(HIDDEN, 10)

        self.fc2normal = nn.Linear(HIDDEN, 10)
        self.fc2negative = nn.Linear(HIDDEN, 10)

        self.net_type = net_type

        self.synergy = 'inactive';

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 30)

        if self.synergy == 'synergy':

            negative = torch.ones_like(x).add(x.neg())

            x1 = self.fc2normal(F.dropout(torch.tanh(self.fc1normal(x)), training=self.training))
            x2 = self.fc2negative(F.dropout(torch.tanh(self.fc1negative(negative)), training=self.training))

            x = x1 + x2

        else:

            if self.net_type == 'negative':
                x = torch.ones_like(x).add(x.neg())
            
            if self.synergy == 'inactive':
                x = self.fc2(F.dropout(torch.tanh(self.fc1(x)), training=self.training))
            elif self.synergy == 'normal':
                x = self.fc2normal(F.dropout(torch.tanh(self.fc1normal(x)), training=self.training))
            elif self.synergy == 'negative':
                x = self.fc2negative(F.dropout(torch.tanh(self.fc1negative(x)), training=self.training))

        return torch.log_softmax(x, dim=1)
