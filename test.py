import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.allLoader import allDataSet
from model.FISTA.DTVSTAnet import DTVNet

device = 0
lossFunction = torch.nn.L1Loss(reduction="mean")
validPathAll = ["path_to_test_data"]
validSet = allDataSet(validPathAll, device)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DTVNet((256,256,64),5).to(device)
dictionary_path = "./checkpoint/dtvnet_v2.5_15.8241200932.dict"
save_root = "result_compare/dtvnet"


save_root_label = "./label_result"

optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
dictionary = torch.load(dictionary_path, map_location=torch.device('cpu'))
net.load_state_dict(dictionary["model"])
optimizer.load_state_dict(dictionary["optimizer"])
scheduler.load_state_dict(dictionary["scheduler"])

net.eval()
with torch.no_grad():
    with tqdm(validLoader) as iterator:
        iterator.set_description("validating...")
        for idx, data in enumerate(iterator):
            input, projection, label, fileName = data
            fileName = fileName[0]
            if "simulation_init_" in fileName:
                file_save_name = os.path.join(save_root, fileName.split("/")[-3], fileName.split("/")[-2])
                file_save_name_label = os.path.join(save_root_label, fileName.split("/")[-3], fileName.split("/")[-2])
            else:
                file_save_name = os.path.join(save_root, fileName.split("/")[-2])
                file_save_name_label = os.path.join(save_root_label, fileName.split("/")[-2])
            if not os.path.exists(file_save_name):
                os.makedirs(file_save_name)
            if not os.path.exists(file_save_name_label):
                os.makedirs(file_save_name_label)
            input, projection, label = input.to(device), projection.to(device), label.to(device)
            tic = time.time()
            output = net(input, projection)
            # output, _ = net(input, projection) # LIRNet
            print("time:{}".format(time.time()-tic))
            loss = lossFunction(output[-1], label)
            output[-1].detach().cpu().numpy().tofile(os.path.join(file_save_name, fileName.split("/")[-1]))
            label.detach().cpu().numpy().tofile(os.path.join(file_save_name_label, fileName.split("/")[-1]))
            iterator.set_postfix_str("loss:{}".format(loss.item()))
