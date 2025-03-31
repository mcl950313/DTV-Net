import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from thop import profile
from options import * 
from model.FISTA.DTVSTAnet import *

device = 0
model = DTVNet((256,256,64),5).to(device)
input1 = torch.randn(1,1,64,256,256).to(device)
input2 = torch.randn(1,1,72*21,144,80).to(device)

flops, params = profile(model, inputs=(input1, input2))

print(f"FLOPs: {flops}, Parameters: {params}")