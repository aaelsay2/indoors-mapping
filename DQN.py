import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500000
EPS_STEP_END = 6000000

def preprocess_frame(frame):
    # frame = torch.from_numpy(frame)
    # frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)

    return frame

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.num_actions = outputs
        self.device = torch.device('cuda')

        self.steps = 0
        hidden_layer = 512

        # Conv Module
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.feature_extraction = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.conv2,
            self.bn2,
            nn.ReLU(),
            self.conv3,
            self.bn3,
            nn.ReLU())

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        # Value Module
        self.state_fc = nn.Linear(linear_input_size, hidden_layer)
        self.state_values = nn.Linear(hidden_layer, 1)

        # Advantage Module
        self.action_fc = nn.Linear(linear_input_size, hidden_layer)
        self.action_values = nn.Linear(hidden_layer, self.num_actions)
        
        # self.head = nn.Linear(linear_input_size, self.num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        normalize_image = False

        if normalize_image:
            x = x / 255     

        # print('Input size: {}'.format(x.size()))
        # x = F.relu(self.bn1(self.conv1(x)))
        # print('1st cnn size: {}'.format(x.size()))
        # x = F.relu(self.bn2(self.conv2(x)))
        # print('2nd cnn size: {}'.format(x.size()))
        # x = F.relu(self.bn3(self.conv3(x)))
        # print('3nd cnn size: {}'.format(x.size()))
        # print(x.size())
        # print('Output size: {}'.format(self.head(x.view(x.size(0), -1)).size()))

        x = self.feature_extraction(x).view(x.size(0), -1)

        # Through Advantage Module
        action_v = self.action_values(self.action_fc(x))

        # Through Value Module
        state_v = self.state_values(self.state_fc(x))

        # return self.head(x.view(x.size(0), -1))
        return state_v + action_v - action_v.mean(dim=1, keepdim=True)

    def _get_eps(self):
        if(self.steps > EPS_STEP_END):
            return 0
        decay_factor = math.exp(-1. * self.steps / EPS_DECAY)
        return EPS_END + (EPS_START - EPS_END) * decay_factor

    def get_greedy_action(self, state, update_step=True):
        '''
        gets action based on e-greedy algorithm
        '''
        if(update_step):
            self.steps += 1
        eps = self._get_eps()
        if(random.random() > eps):
            # state = preprocess_frame(state)
            return self.get_action(state)
        else:
            # random_actions = random.randrange(self.num_actions)
            # return random_actions.view(1, 1)
            return torch.LongTensor(1).random_(self.num_actions).to(self.device).view(1, 1)

    def get_action(self, state):
        # dstate = torch.tensor(state).to(self.device)
        dstate = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.forward(dstate).max(1)[1].view(1, 1) # .item()

# def select_action(state, update_step=True):
#     global steps_done
#     sample = random.random()
#     eps_threshold = _get_eps()
#     if update_step:
#       steps_done += 1

#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             state = preprocess_frame(state)
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.LongTensor(1).random_(n_actions).to(device).view(1, 1) #torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
