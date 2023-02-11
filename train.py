"""
train reference:
https://colab.research.google.com/drive/1ryOQ6hXxCidM_mGN0Yrf4BbjUtpyCxgy#scrollTo=bKPrbgel8xr-
"""

import sys
from PKGNN import PKGNN_v1
from utils import get_fe_crystal, get_ts, create_F_from_input
import torch
from tqdm import tqdm
from data import get_dataloader
from torch.optim.lr_scheduler import StepLR
from math import inf
import os

# now let's train our model!
# import wandb
# wandb.init(project="PK GNN v1", reinit=True, anonymous="must")
# wandb.watch(model)
# wandb.log({"Epoch": epoch,"Train_Loss": loss,})

# prepare data
batch_size = 64
train_loader, test_loader = get_dataloader(batch_size = batch_size)

# prepare ckpt path
# time_stamp = get_ts()
# use shell timestamp 
time_stamp = sys.argv[1]
ckpt_path = f'checkpoints/{time_stamp}/'

# set hyper params
lr = 1e-1
num_epochs = 60

# init model and optimizer
model = PKGNN_v1(get_fe_crystal())
model.to('cuda')
# show model params: model.named_parameters():
optim = torch.optim.Adam(model.parameters(), lr=lr)
optim.zero_grad()
scheduler = StepLR(optim, step_size=30, gamma=0.3)

best_eval_loss = inf
all_loss = []
# start training
for epoch in range(num_epochs):
    # train
    for F_00, F_01, s_true in tqdm(train_loader):
        F = create_F_from_input(F_00, F_01)
        s_pred = model(F.to('cuda'))

        err: torch.Tensor = (s_pred - s_true.to('cuda')).abs()

        loss = err.mean() * 1e5 # .pow(2)
        all_loss.append(float(loss.detach()))
        """
        add regularization: 
        https://androidkt.com/how-to-add-l1-l2-regularization-in-pytorch-loss-function/
        loss = err.mean() * 1e5 + sum(p.pow(2).sum() for p in model.parameters())
        """

        loss.backward()
        optim.step()
        optim.zero_grad()
        break
    scheduler.step()

    # test
    with torch.no_grad():
        err = torch.zeros(9, dtype = torch.float32)
        for F_00, F_01, s_true in tqdm(test_loader):
            F = create_F_from_input(F_00, F_01)
            s_pred = model(F.to('cuda'))
            err += (s_pred.cpu() - s_true).abs().mean(0)
        err /= len(test_loader)
        log_err = torch.log10(err)
        loss = err.mean() * 1e4
        # print("sample s_pred = ", s_pred[0, :2])
        # print("sample s_true = ", s_true[0, :2])
        print("mean err = ", err)
        print("loss = ", loss)

        """saving models: https://discuss.pytorch.org/t/how-to-save-the-best-model/84608"""
        if best_eval_loss > loss.detach():
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(model.state_dict(), ckpt_path + f'best-model-params.pt')
            best_eval_loss = loss.detach()
    
    break

with open(ckpt_path + 'loss.txt', 'w') as f:
    for val in loss:
        f.write(str(val) + "\n")