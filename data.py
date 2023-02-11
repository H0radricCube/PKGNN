"""
dataset & dataloader & visualize
"""

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt


class PK_Dataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        data = np.loadtxt(path, dtype=np.float32)
        self.n_samples = data.shape[0]
        self.F_00 = data[:, 0]
        self.F_01 = data[:, 1]
        self.stress = data[:, 2:]

    def __getitem__(self, index):
        return self.F_00[index], self.F_01[index], self.stress[index]

    def __len__(self):
        # FIXME
        return self.n_samples


def get_dataloader(path='data/strain-stress-uniform.dat', batch_size=64):
    dataset = PK_Dataset(path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def draw_hotmap(data, title, tick_num=9, save=None):
    """
    draw DPK style hotmap
    data need to be preprocess data to proper shape and order

    usage:
    dataset = PK_Dataset('data/strain-stress-uniform.dat')
    data = dataset.stress[:, 0].reshape(100, 100).T[::-1, :]
    draw_hotmap(data, title = "P_11 by viral formula")
    """
    x_size, y_size = data.shape[1], data.shape[0]
    x_tick_pos = np.linspace(0, x_size, tick_num, endpoint=True)
    y_tick_pos = np.linspace(y_size, 0, tick_num, endpoint=True)
    x_tick_label = [f'{val:.3f}' for val in np.linspace(0.9, 1.1, tick_num, endpoint=True)]
    y_tick_label = [f'{val:.3f}' for val in np.linspace(-0.1, 0.1, tick_num, endpoint=True)]

    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='jet', vmin=-0.015, vmax=0.015)
    plt.xticks(x_tick_pos, x_tick_label)
    plt.xlabel("F11")
    plt.yticks(y_tick_pos, y_tick_label)
    plt.ylabel("F12")
    plt.colorbar(location='bottom', shrink=0.73, pad=0.1)
    plt.title(title)

    ax = plt.gca()
    if save is None:
        plt.show()
    else:
        plt.savefig(save, facecolor = 'white')
