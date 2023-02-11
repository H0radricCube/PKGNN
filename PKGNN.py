from e3nn import o3
import torch
from message_passing import MessagePassing_v1
from utils import create_graph, Q_pos


class PKGNN_v1(torch.nn.Module):
    """
    reference: e3nn\e3nn\nn\models\v2106\gate_points_networks.py
    the simplest version with no spherical_harmonics
    no node attr
    """

    def __init__(
        self,
        crystal,
        irreps_node="1o",
        max_edge_type=3,
        mul=5,
        layers=2,
        lmax=2,  # 用于构造 irreps_node_hidden
    ) -> None:
        super().__init__()

        edge_src, edge_dst, node_pos, edge_type, self.encoder = create_graph(
            crystal, max_edge_type=max_edge_type)
        
        # node_pos: (n, 3, 1)
        self.edge_src = torch.nn.Parameter(edge_src, requires_grad = False)
        self.edge_dst = torch.nn.Parameter(edge_dst, requires_grad = False)
        self.node_pos = torch.nn.Parameter(node_pos, requires_grad = False)
        self.edge_type = torch.nn.Parameter(edge_type, requires_grad = False)

        num_nodes = self.node_pos.shape[0]
        num_neighbors = len(self.edge_src) / num_nodes
        self.num_nodes = num_nodes

        irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing_v1(
            irreps_node_sequence=[irreps_node] + layers * [irreps_node_hidden] + [irreps_node],
            fc_neurons=[max_edge_type, 20],
            num_neighbors=num_neighbors,
        )
        self.w = torch.nn.Parameter(torch.rand(num_nodes * num_nodes))

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """ F:deformation gradient"""
        
        batch_size = F.shape[0]

        # F: (b, 3, 3), node_pos: (n, 3) -> Fx: (b, n, 3)
        Fx = torch.einsum("bij,nj->bni", F, self.node_pos)
        Fx = self.mp(
            self.edge_src, self.edge_dst, 
            Fx, self.edge_type)

        src_emb = Fx[:, self.edge_src]
        dst_emb = Fx[:, self.edge_dst]
        # Fx_ij & x_ij: (b, e, 3)
        Fx_ij = src_emb - dst_emb
        x_ij = torch.einsum("bij,bej->bei", torch.linalg.inv(F), Fx_ij)
        tp_res_list = []

        """
        HACK: simple tensor product
        both torch.kron and torch.einsum('ni,mj->nmij') give unexpected zeros
        """
        for b in range(batch_size):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    tp_res_list.append(torch.kron(Fx_ij[b, i], x_ij[b, j]))

        # tp_res: (b, e^2, 9)
        tp_res = torch.stack(tp_res_list).view(batch_size, -1, 9)

        # weighted sum of stress tensors: (b, 9)
        stress = torch.einsum('n,bni->bi', self.w, tp_res)

        return stress


if __name__ == '__main__':
    from utils import get_fe_crystal, Q_pos
    import numpy as np

    F = np.array([
        [2, -1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    F = torch.from_numpy(np.stack([F, F]))

    # check equivariance: https://docs.e3nn.org/en/latest/guide/convolution.html

    rot = o3.rand_matrix()
    model = PKGNN_v1(get_fe_crystal())

    # rotate before
    # f_before = conv(f_in @ D_in.T, pos @ rot.T)
    f_before = model(rot @ F).reshape(-1, 3, 3)

    # rotate after
    f_after = rot @ model(F).reshape(-1, 3, 3)

    print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))
