from e3nn import o3
import torch
from utils import Compose, tp_path_exists
from conv import Convolution_v1
from e3nn.nn import Gate

act = {
    1: torch.nn.functional.silu,
    -1: torch.tanh,
}

act_gates = {
    1: torch.sigmoid,
    -1: torch.tanh,
}


class MessagePassing_v1(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_sequence : list of `e3nn.o3.Irreps`
        representation of the input/hidden/output features
        (without concatenating node pos)

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_sequence,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        irreps_node_sequence = [o3.Irreps(irreps) for irreps in irreps_node_sequence]

        self.layers = torch.nn.ModuleList()

        self.layer_sizes = [irreps_node_sequence[0]]  # this is for brief demonstrating the model

        irreps_node = irreps_node_sequence[0]
        mid_irreps = o3.Irreps()

        """
        gate construction
        https://docs.e3nn.org/en/latest/api/nn/nn_gate.html
        """
        for irreps_node_hidden in irreps_node_sequence[1:-1]:
            # 1x0e
            irreps_scalars = o3.Irreps(
                [(mul, ir) for mul, ir in
                 irreps_node_hidden if
                 ir.l == 0 and tp_path_exists(
                    self.concated_irreps(irreps_node, mid_irreps),
                    self.concated_irreps(irreps_node, mid_irreps), ir)]
            ).simplify()

            # 1x1e+1x2e
            irreps_gated = o3.Irreps(
                [(mul, ir) for mul, ir in
                 irreps_node_hidden
                 if ir.l > 0 and tp_path_exists(
                    self.concated_irreps(irreps_node, mid_irreps),
                    self.concated_irreps(irreps_node, mid_irreps), ir)]
            )
            if irreps_gated.dim > 0:
                if tp_path_exists(
                        self.concated_irreps(irreps_node, mid_irreps),
                        self.concated_irreps(irreps_node, mid_irreps),
                        "0e"):
                    ir = "0e"
                elif tp_path_exists(
                        self.concated_irreps(irreps_node, mid_irreps),
                        self.concated_irreps(irreps_node, mid_irreps),
                        "0o"):
                    ir = "0o"
                else:
                    raise ValueError("Unable to produce gates")
                    # raise ValueError(
                    #     f"irreps_node={irreps_node} times irreps_edge_attr={self.irreps_edge_attr} is unable to produce gates "
                    #     f"needed for irreps_gated={irreps_gated}")
            else:
                ir = None
            # 0e
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution_v1(
                self.concated_irreps(irreps_node, mid_irreps),
                gate.irreps_in, fc_neurons, num_neighbors)

            self.layers.append(Compose(conv, gate))
            mid_irreps = gate.irreps_out  # update the irreps_input of next conv layer
            self.layer_sizes.append(self.concated_irreps(irreps_node, mid_irreps))

        irreps_node_output = irreps_node_sequence[-1]
        self.layers.append(
            Convolution_v1(
                self.concated_irreps(irreps_node, mid_irreps),
                irreps_node_output, fc_neurons, num_neighbors)
        )
        self.layer_sizes.append(irreps_node_output)

    # HACK: 两个 concat 顺序必须一致

    @staticmethod
    def concated_irreps(irreps_node, mid_irreps):
        return irreps_node + mid_irreps

    @staticmethod
    def concated_tensor(node_pos, mid_emb):
        return torch.cat([node_pos, mid_emb], dim=-1)

    def forward(self, edge_src, edge_dst, node_pos, edge_type) -> torch.Tensor:

        # node_pos: (b, n, 3)
        mid_emb = node_pos
        for lay in self.layers[:-1]:
            # mid_emb: (b, n, x)
            mid_emb = lay(edge_src, edge_dst, mid_emb, edge_type)
            # mid_emb: (b, n, 3 + x)
            mid_emb = self.concated_tensor(node_pos, mid_emb)
        F_output = self.layers[-1](edge_src, edge_dst, mid_emb, edge_type)

        return F_output

    @property
    def irreps_in(self):
        return self.layer_sizes[0]

    @property
    def irreps_out(self):
        return self.layer_sizes[-1]


if __name__ == '__main__':
    from utils import get_fe_crystal, create_graph, Q_pos

    edge_src, edge_dst, node_pos, edge_type, encoder = create_graph(get_fe_crystal())

    # check equivariance: https://docs.e3nn.org/en/latest/guide/convolution.html

    rot = o3.rand_matrix()
    irreps_input = "1o"

    irreps_node_hidden = o3.Irreps([(10, (l, p)) for l in range(3) for p in [-1, 1]])

    fc_neurons = [edge_type.shape[1]]
    num_neighbors = len(edge_src) / len(node_pos)
    irreps_node_sequence = [irreps_input] + 2 * [irreps_node_hidden] + [irreps_input]

    mp = MessagePassing_v1(irreps_node_sequence, fc_neurons, num_neighbors)

    # fake batched node_pos
    node_pos = node_pos[None, ...]

    # rotate before
    f_before = mp(edge_src, edge_dst, torch.einsum("ij,bnj->bni", rot, node_pos), edge_type)

    # rotate after
    f_after = mp(edge_src, edge_dst, node_pos, edge_type)
    f_after = torch.einsum("ij,bnj->bni", rot, f_after)

    print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))
