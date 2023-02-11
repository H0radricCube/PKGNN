import sys
import torch
from utils import load_model, get_fe_crystal, create_F_from_input
from PKGNN import PKGNN_v1
from data import get_dataloader
from tqdm import tqdm
from itertools import chain
import numpy as np

time_stamp = sys.argv[1]
ckpt_path = f'checkpoints/{time_stamp}/'


# draw map

# load model
state_dict_path = ckpt_path + 'best-model-params.pt'
model = load_model(PKGNN_v1, state_dict_path, crystal = get_fe_crystal())
batch_size = 64

# load data
train_loader, test_loader = get_dataloader(batch_size = batch_size)

x = []
y = []

P00_pred = []
P01_pred = []

print("start eval...")
with torch.no_grad():
    err = torch.zeros(9, dtype = torch.float32)
    for F_00, F_01, s_true in tqdm(chain(train_loader, test_loader)):
        F = create_F_from_input(F_00, F_01)
        s_pred = model(F)
        # 暂时不存 
        # x.append(F_00)
        # y.append(F_01)
        P00_pred.append(s_pred[:, 0])
        P01_pred.append(s_pred[:, 1])
        err += (s_pred - s_true).abs().mean(0)
    err = torch.log10(err / len(test_loader))

# res save path 
res_path = ckpt_path + 'res.npz'
# x = torch.concat(x).numpy()
# y = torch.concat(y).numpy()
P00_pred = torch.concat(P00_pred).numpy()
P01_pred = torch.concat(P01_pred).numpy()

print("eval_err = ", err)

with open(res_path, 'wb') as f:
    np.savez(
        f,
        P00_pred = P00_pred, 
        P01_pred = P01_pred,
        err = err
    )

