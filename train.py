import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.allLoader import allDataSet
from model.FISTA.DTVSTAnet import DTVNet
import setproctitle
setproctitle.setproctitle("DTV-net-mcl-submit")
lossFunction = torch.nn.L1Loss(reduction="mean")

device = 0
trainPathAll = ["path_to_train_data"]
validPathAll = ["path_to_val_data"]
trainSet = allDataSet(trainPathAll, device)
validSet = allDataSet(validPathAll, device)
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False)

net = DTVNet((256,256,64),5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

# dictionary = torch.load("./checkpoint/dtvnet_v2.5_20.7151390792.dict", map_location=torch.device('cpu'))
# net.load_state_dict(dictionary["model"])
# optimizer.load_state_dict(dictionary["optimizer"])
# scheduler.load_state_dict(dictionary["scheduler"])

epoch = 31

for i in range(epoch):
    trainLoss = []
    validLoss = []
    net.train()
    with tqdm(trainLoader) as iterator:
        iterator.set_description("Epoch {}".format(i))
        for idx,data in enumerate(iterator):
            input, projection, label, _ = data
            input, projection, label = input.to(device), projection.to(device), label.to(device)
            output = net(input, projection)
            optimizer.zero_grad()
            loss =lossFunction(output[-1], label)
            loss.backward()
            optimizer.step()
            trainLoss.append(loss.item())
            iterator.set_postfix_str(
                "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(trainLoss))))
    
    net.eval()
    with torch.no_grad():
        with tqdm(validLoader) as iterator:
            iterator.set_description("validating...")
            for idx, data in enumerate(iterator):
                input, projection, label, _ = data
                input, projection, label = input.to(device), projection.to(device), label.to(device)
                output = net(input, projection)
                loss = lossFunction(output[-1], label)
                validLoss.append(loss.item())
                iterator.set_postfix_str(
                    "loss:{},epoch mean:{:.2f}".format(loss.item(), np.mean(np.array(validLoss))))
    
    scheduler.step()
    if i%1==0: torch.save({
        "epoch": i, "model": net.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()
    }, "{}/dtvnet_v2.5_{:.10f}.dict".format("./checkpoint", np.mean(np.array(trainLoss))))
