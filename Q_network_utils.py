import torch
import torch.sparse as sparse
from torch import Tensor

initialization_stddev = 0.01


def tensor_init(shape: list | tuple, tensor_type, method="truncated normal") -> Tensor:
    if method == "truncated normal":
        t = torch.ones(shape, dtype=tensor_type)
        t = torch.nn.init.trunc_normal_(
            t,
            std=initialization_stddev,
            a=(-2.0) * initialization_stddev,
            b=2.0 * initialization_stddev
        )

    else:
        raise Exception("No method called \'" + str(method) + "\'")

    return t


def get_graph_reconstruction_loss(
        h_v_l: Tensor,
        laplacian_node_node: sparse.Tensor,
        node_node: sparse.Tensor
):
    temp = 2 * torch.trace(torch.matmul(
        h_v_l.T,
        sparse.mm(
            laplacian_node_node,
            h_v_l
        )
    ))

    edge_num = torch.sum(node_node)
    return torch.divide(temp, edge_num)
