import torch
import torch_geometric
import ase
from sklearn.preprocessing import OneHotEncoder
import ase.neighborlist
import numpy as np
from e3nn import o3
from datetime import datetime


def Q_pos(Q, node_pos):
    """
    used explicity for multiple node pos
    not contiguous: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch

    node_pos: (n, 3, 1)
    Q: ((b,) 3, 3)
    """
    return torch.t(torch.matmul(Q, torch.t(node_pos)))


"""
1x1o x 1x1o -> 1x0e+1x1e+1x2e
1x1e x 1x1o -> 1x0o+1x1o+1x2o
"""


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def get_fe_crystal():
    """
    create fe unit cell
    reference: https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html
    """
    fe_lattice = torch.eye(3)
    fe_coords = (torch.tensor([
        [0., 0., 0.],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0.5, 0.5, 0.5]
    ]) - 0.5)
    fe_types = ['Fe'] * 9
    fe = ase.Atoms(symbols=fe_types, positions=fe_coords, cell=fe_lattice, pbc=False)
    return fe


def create_graph(crystal, radial_cutoff=1.1, max_edge_type=5, encoder=None):
    """
    dataset creation:

    reference: https://docs.e3nn.org/en/latest/guide/periodic_boundary_conditions.html
    radial_cutoff: create edges that is exactly smaller than this cutoff

    one_hot_edge encoding edge type by edge length
    """

    """
    ase.neighborlist.neighbor_list:

    edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    edge_shift indicates whether the neighbors are in different images / copies of the unit cell
    reference https://databases.fysik.dtu.dk/ase/ase/neighborlist.html

    need to import ase.neighborlist
    """
    edge_src, edge_dst = ase.neighborlist.neighbor_list(
        "ij", a=crystal, cutoff=radial_cutoff, self_interaction=True)  # 加入自环，在 aggregate 的时候不用自己加上自己了

    """
    torch_geometric.data.Data

    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
    """
    graph = torch_geometric.data.Data(
        pos=torch.tensor(crystal.get_positions(), dtype=torch.float32),
        # lattice=torch.tensor(crystal.cell.array).unsqueeze(0),  # We add a dimension for batching
        # edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        x=None,  # node features
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
    )

    # dataset.edge_index and graph.pos is what we need
    edge_src = graph.edge_index[0]
    edge_dst = graph.edge_index[1]

    # fit one-hot encoder of edge types by length
    edge_vec = graph["pos"][edge_src] - graph["pos"][edge_dst]
    edge_length = edge_vec.norm(dim=1)

    """
    One Hot Encoder
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, dtype=np.float32)
        encoder.fit(edge_length[:, None])

    one_hot_edge = encoder.transform(edge_length[:, None])
    pad_size = max_edge_type - one_hot_edge.shape[1]
    assert(pad_size >= 0)
    one_hot_edge = torch.nn.functional.pad(torch.from_numpy(one_hot_edge), (0, pad_size), 'constant', 0)

    return edge_src, edge_dst, graph["pos"], one_hot_edge, encoder


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


def get_ts():
    return datetime.now().strftime("%Y%m%d-%H-%M")


base_F = torch.eye(3, dtype=torch.float32)


def create_F_from_input(F_00, F_01):
    F = torch.tile(base_F, (F_01.shape[0], 1, 1))
    F[:, 0, 0] += F_00
    F[:, 0, 1] += F_01
    return F

def load_model(MODEL, state_dict_path, **model_kwargs):
    model = MODEL(**model_kwargs)
    model.load_state_dict(torch.load(state_dict_path))
    return model

if __name__ == '__main__':
    import numpy as np

    F = np.array([
        [2, -1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    F = torch.from_numpy(F)
    inv_F = torch.linalg.inv(F)

    _, _, node_pos, edge_vec = create_graph(get_fe_crystal())
    print(edge_vec)

    edge_length = edge_vec.norm(dim=1)

    encoder = OneHotEncoder(sparse=False)
    max_categories = 5
    output = encoder.fit_transform(edge_length[:, None])
    pad_size = max_categories - output.shape[1]
    assert(pad_size >= 0)
    output = torch.nn.functional.pad(torch.from_numpy(output), (0, pad_size), 'constant', 0)
    print(output)
