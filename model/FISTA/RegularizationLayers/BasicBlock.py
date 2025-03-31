import numpy as np
import torch
import torch.nn as nn

class BasicBlockDTV_AE(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features=32):
        super(BasicBlockDTV_AE, self).__init__()
        self.Sp = nn.Softplus()
        self.in_out_channel = 32
        self.conv1_forward = nn.Conv3d(self.in_out_channel, features, (3, 3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv3d(features, features, (3, 3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv3d(features, self.in_out_channel, (3, 3, 3), stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.Sp = nn.Softplus()
        self.mu = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

    def forward(self, x):
        x_input = x  

        l1 = self.conv1_forward(x_input) 
        l1 = self.relu(l1)
        l2 = self.conv2_forward(l1) 
        l2 = self.relu(l2)
        l3 = self.conv3_forward(l2) 
        l3 = self.relu(l3)
        x_forward = self.relu(self.conv4_forward(l3)) 

        x_forward = torch.mul(torch.sign(x_forward), self.relu(torch.abs(x_forward) - self.Sp(self.mu)))
        
        l1b = self.conv1_backward(x_forward) 
        l1b = self.relu(l1b)

        l2b = self.conv2_backward(l1b) 
        l2b = self.relu(l2b)

        l3b = self.conv3_backward(l2b)
        l3b = self.relu(l3b)

        x_backward = self.conv4_backward(l3b)
        # x_backward = self.relu(x_backward)

        x_pred = x_input + x_backward

        return x_pred

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)

class HAB(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features=8):
        super(HAB, self).__init__()
        self.conv3d1 = nn.Conv3d(in_channels=1,out_channels=features, kernel_size=3, padding=1)
        self.conv3d2 = nn.Conv3d(in_channels=features,out_channels=features, kernel_size=3, padding=1)
        self.conv3d3 = nn.Conv3d(in_channels=features,out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1_input, x2_input):
        x1 = self.conv3d1(x1_input)
        x2 = self.conv3d2(x2_input)
        x = self.sigmoid(x1 + x2)
        x_output = x * x2_input
        x_output = self.conv3d3(x_output)

        return x_output

