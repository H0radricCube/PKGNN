import numpy as np
import sys
import matplotlib.pyplot as plt
from data import draw_hotmap

time_stamp = sys.argv[1]
ckpt_path = f'checkpoints/{time_stamp}/'

res_path = ckpt_path + 'res.npz'

res = np.load(res_path)

draw_hotmap(
    res['P00_pred'].reshape(100, 100).T[::-1, :], 
    title = "P_11 by GNN", save = ckpt_path + 'P_11_GNN.png')

draw_hotmap(
    res['P01_pred'].reshape(100, 100).T[::-1, :], 
    title = "P_12 by GNN", save = ckpt_path + 'P_12_GNN.png')
