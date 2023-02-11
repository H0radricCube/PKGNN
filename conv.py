import torch
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, FullTensorProduct
from e3nn import o3
import matplotlib.pyplot as plt

from torch_scatter import scatter

"""
tensor products:

reference: https://docs.e3nn.org/en/latest/api/o3/o3_tp.html
FullTensorProduct the natural full tensor product
FullyConnectedTensorProduct: weighted sum of compatible paths
"""

"""
def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)
"""

"""equivariant convolution vanilla: https://docs.e3nn.org/en/latest/guide/convolution.html"""

# this doesn't support multilayer properly


class Convolution_v0(torch.nn.Module):
    r"""
    Parameters
    ----------
    irreps_input : `e3nn.o3.Irreps`
        representation of the node

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_input, irreps_output, fc_neurons, num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.num_neighbors = num_neighbors
        self.fc_neurons = fc_neurons

        self.tp = FullTensorProduct(
            irreps_input,
            irreps_input
        )

        tp2 = FullyConnectedTensorProduct(
            self.irreps_input,
            self.tp.irreps_out,
            self.irreps_input,
            internal_weights=False,
            shared_weights=False
        )
        self.tp2 = tp2

        self.fc = FullyConnectedNet(fc_neurons + [tp2.weight_numel], torch.nn.functional.silu)

    def forward(self, edge_src, edge_dst, node_pos, edge_type) -> torch.Tensor:

        assert(edge_type.shape[1] == self.fc_neurons[0])
        weight = self.fc(edge_type)

        mix = self.tp(node_pos[edge_src], node_pos[edge_dst])
        edge_message = self.tp2(node_pos[edge_src], mix, weight)

        node_conv_out = scatter(edge_message, edge_dst, dim_size=node_pos.shape[0]).div(self.num_neighbors**0.5)

        return node_conv_out


class Convolution_v1(torch.nn.Module):
    r"""
    Parameters
    ----------
    irreps_input : `e3nn.o3.Irreps`
        representation of the node

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_input, irreps_output, fc_neurons, num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.num_neighbors = num_neighbors
        self.fc_neurons = fc_neurons

        tp = FullyConnectedTensorProduct(
            self.irreps_input,
            self.irreps_input,
            self.irreps_output,
            internal_weights=False,
            shared_weights=False
        )
        self.tp = tp
        # fig, ax = tp.visualize()
        # plt.show()
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)

    def forward(self, edge_src, edge_dst, node_emb, edge_type) -> torch.Tensor:
        """
        node_emb: (b, n, 3)
        """

        assert(edge_type.shape[1] == self.fc_neurons[0])
        weight = self.fc(edge_type)

        src_emb = node_emb[:, edge_src]
        dst_emb = node_emb[:, edge_dst]
        
        # edge_message: (b, e, 3)
        edge_message = self.tp(src_emb, dst_emb, weight)

        # node_conv_out: (b, n, 3)
        node_conv_out = scatter(edge_message, edge_dst, dim = 1, reduce="sum").div(self.num_neighbors**0.5)

        return node_conv_out


if __name__ == '__main__':
    from utils import get_fe_crystal, create_graph, Q_pos

    edge_src, edge_dst, node_pos, edge_type, encoder = create_graph(get_fe_crystal())

    # check equivariance: https://docs.e3nn.org/en/latest/guide/convolution.html

    rot = o3.rand_matrix()
    irreps_input = "1o"
    fc_neurons = [edge_type.shape[1]]
    num_neighbors = len(edge_src) / len(node_pos)

    conv = Convolution_v1(irreps_input, irreps_input, fc_neurons, num_neighbors)

    # fake batched node_pos
    node_pos = node_pos[None, ...]

    # rotate before
    f_before = conv(edge_src, edge_dst, torch.einsum("ij,bnj->bni", rot, node_pos), edge_type)
    # rotate after
    f_after = conv(edge_src, edge_dst, node_pos, edge_type)
    f_after = torch.einsum("ij,bnj->bni", rot, f_after)

    print(torch.allclose(f_before, f_after, rtol=1e-4, atol=1e-4))
