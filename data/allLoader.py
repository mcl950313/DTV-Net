import os
import numpy as np
import torch
from torch.utils.data import Dataset
from model.ConeBeamLayers.Beijing.BeijingGeometry import ForwardProjection, BeijingGeometryWithFBP

class allDataSet(Dataset):
    def __init__(self, dirs, device, scale=(1,64,256,256)):
        self.dir = []
        self.num_upper = 10000
        for dir in dirs:
            img_dir_path = os.path.join(dirs, dir)
                self.dir.append(os.path.join(img_dir_path)
        self.scale = scale
        self.device = device
        self.fdk = BeijingGeometryWithFBP().to(self.device)

    def __getitem__(self,index):
        img = np.fromfile(self.dir[index], "float32")
        img = np.reshape(img, self.scale)
        img = torch.unsqueeze(torch.from_numpy(img),0).to(self.device)
        proj = ForwardProjection.apply(img)
        input = self.fdk(proj)
        input = input.reshape(self.scale).cpu()
        img = torch.squeeze(img, 0).cpu()
        proj = torch.squeeze(proj, 0).cpu()
        return input, proj, img, self.dir[index]

    def __len__(self):
        return len(self.dir)

