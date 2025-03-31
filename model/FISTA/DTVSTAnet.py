import torch
import torch.nn as nn
import numpy as np
from model.ConeBeamLayers.Beijing.BeijingGeometry import *
from model.PDHG.Utils.Gradients import spraseMatrixX, spraseMatrixY, spraseMatrixZ
from model.FISTA.RegularizationLayers.BasicBlock import BasicBlockDTV_AE, HAB

    
class DTVNet(nn.Module):
    def __init__(self, volumeSize, cascades: int = 5):
        super(DTVNet, self).__init__()
        self.cascades = cascades
        self.ITE = BeijingGeometry()
        self.fdk = BeijingGeometryWithFBP()
        self.block = BasicBlockDTV_AE()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.channel = 8
        self.conv3d1 = nn.Conv3d(in_channels=1,out_channels=self.channel, kernel_size=3, padding=1)
        self.conv3d2 = nn.Conv3d(in_channels=1,out_channels=self.channel, kernel_size=3, padding=1)
        self.conv3d3 = nn.Conv3d(in_channels=1,out_channels=self.channel, kernel_size=3, padding=1)
        self.conv3d4 = nn.Conv3d(in_channels=1,out_channels=self.channel, kernel_size=3, padding=1)

        self.HAB_p = HAB()
        self.HAB_q = HAB()
        self.HAB_s = HAB()
        self.HAB_z = HAB()
        self.dx, self.dxt, normDx = spraseMatrixX(volumeSize)
        self.dx, self.dxt = nn.Parameter(self.dx, requires_grad=False), nn.Parameter(self.dxt, requires_grad=False)
        self.dy, self.dyt, normDy = spraseMatrixY(volumeSize)
        self.dy, self.dyt = nn.Parameter(self.dy, requires_grad=False), nn.Parameter(self.dyt, requires_grad=False)
        self.dz, self.dzt, normDz = spraseMatrixZ(volumeSize)
        self.dz, self.dzt = nn.Parameter(self.dz, requires_grad=False), nn.Parameter(self.dzt, requires_grad=False)
        self.ntx = nn.Parameter(torch.tensor([-0.01] * cascades), requires_grad=True)
        self.nty = nn.Parameter(torch.tensor([-0.01] * cascades), requires_grad=True)
        self.ntz = nn.Parameter(torch.tensor([-0.01] * cascades), requires_grad=True)
        self.nt = nn.Parameter(torch.tensor([-0.1] * cascades), requires_grad=True)
        self.lammbda = nn.Parameter(torch.tensor([1.0] * cascades), requires_grad=True)

    def forward(self, image, sino):
        t = [torch.tensor(0)] * (self.cascades + 1)
        t[0] = image
        p = q = s = 0
        for cascade in range(self.cascades):
            res = sino - ForwardProjection.apply(t[cascade])
            z = t[cascade] + self.lammbda[cascade]*self.fdk(res)
            pnew = self.__getGradient(z, self.dx)
            qnew = self.__getGradient(z, self.dy)
            snew = self.__getGradient(z, self.dz)
            pnew_input = self.relu(self.conv3d1(pnew))
            qnew_input = self.relu(self.conv3d2(qnew))
            snew_input = self.relu(self.conv3d3(snew))
            z_input = self.relu(self.conv3d4(z))
            output = self.block(torch.concat((pnew_input, qnew_input, snew_input, z_input), dim=1))

            p_ = self.HAB_p(pnew, output[:,:self.channel])
            q_ = self.HAB_q(qnew, output[:,self.channel:self.channel*2])
            s_ = self.HAB_s(snew, output[:,self.channel*2:self.channel*3])
            znew = self.HAB_z(z, output[:,self.channel*3:]) + z
            
            p = p + self.ntx[cascade] * (p - p_)
            q = q + self.nty[cascade] * (q - q_)
            s = s + self.ntz[cascade] * (s - s_)
            z_ = t[cascade] + self.nt[cascade] * (t[cascade] - znew) 

            p_ = self.__getGradient(p, self.dxt)
            q_ = self.__getGradient(q, self.dyt)
            s_ = self.__getGradient(s, self.dzt)
            t[cascade+1] = q_ + p_ + s_ + z_
        return t

    def __getGradient(self, image, sparse):
        result = []
        for batch in image:
            result.append(torch.reshape(torch.matmul(sparse, batch.view(-1)), batch.size()))
        return torch.stack(result, 0)
    
