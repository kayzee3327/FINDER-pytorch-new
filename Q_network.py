import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.sparse as sparse

from Q_network_utils import tensor_init, get_graph_reconstruction_loss


class QNetwork(nn.Module):
    def __init__(self, input_size, embedding_size, k, tensor_type, device):
        super().__init__()

        self.encoding = Encoding(input_size, embedding_size, k, tensor_type, device)
        self.decoding = Decoding(embedding_size, tensor_type, device)

    def forward(
            self,
            actions_or_node_batch: sparse.Tensor,
            node_node: sparse.Tensor,
            batch_node: sparse.Tensor,
            need_q_for_all: bool,
            need_graph_recons_loss: bool = False,
            laplacian_node_node: sparse.Tensor = None,
            depth=None,
            input_features=None,
            input_feature_s=None
    ):
        if need_graph_recons_loss:
            assert laplacian_node_node is not None
        node_embedding, state_embedding = self.encoding(node_node, batch_node, depth, input_features, input_feature_s)
        Q = self.decoding(actions_or_node_batch, node_embedding, state_embedding, need_q_for_all)

        if need_graph_recons_loss:
            return Q, get_graph_reconstruction_loss(node_embedding, laplacian_node_node, node_node)
        else:
            return Q


class Encoding(nn.Module):
    def __init__(self, input_size, embedding_size, k, tensor_type, device):
        super().__init__()

        # parameters
        self.W_1 = nn.Parameter(tensor_init([input_size, embedding_size], tensor_type, device))
        self.W_2 = nn.Parameter(tensor_init([embedding_size, embedding_size], tensor_type, device))
        self.W_3 = nn.Parameter(tensor_init([embedding_size, embedding_size], tensor_type, device))
        self.relu = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size, device=device),
            nn.ReLU()
        )
        # feature of virtual node s is all one because "Note that Xs contains all-one vectors of size c, which leads
        # the state embedding to be determined by the graph structure only"
        self.input_size = input_size
        self.tensor_type = tensor_type
        self.device = device
        self.depth = k

    def forward(
            self,
            node_node: sparse.Tensor,  # graph representation: node-node
            batch_node: sparse.Tensor,  # graph representation: batch-node
            depth=None,
            input_features=None,
            input_feature_s=None
    ):
        if depth is None:
            if self.depth is None:
                raise TypeError("depth of GraphSAGE is None")
            else:
                depth = self.depth

        node_cnt = node_node.shape[0]
        batch_size = batch_node.shape[0]
        if input_features is None:
            input_features = torch.ones([node_cnt, self.input_size], dtype=self.tensor_type, device=self.device)
        if input_feature_s is None:
            input_feature_s = torch.ones([batch_size, self.input_size], dtype=self.tensor_type, device=self.device)

        # _s suffix means this tensor is related to the virtual node
        h_v_0 = f.relu(torch.matmul(input_features, self.W_1))
        h_v_0_s = f.relu(torch.matmul(input_feature_s, self.W_1))
        h_v_0 = f.normalize(h_v_0)
        h_v_0_s = f.normalize(h_v_0_s)

        # l-1 is omitted below
        h_v_l = h_v_0
        h_v_l_s = h_v_0_s
        for i in range(depth):
            h_v = h_v_l
            h_v_s = h_v_l_s
            h_nv = sparse.mm(node_node, h_v)
            h_nv_s = sparse.mm(batch_node, h_v)
            h_v_l = self.relu(torch.concat(
                [
                    torch.matmul(h_v, self.W_2),
                    torch.matmul(h_nv, self.W_3)
                ],
                dim=1
            ))

            h_v_l_s = self.relu(torch.concat(
                [
                    torch.matmul(h_v_s, self.W_2),
                    torch.matmul(h_nv_s, self.W_3)
                ],
                dim=1
            ))

            h_v_l = f.normalize(h_v_l)
            h_v_l_s = f.normalize(h_v_l_s)

        return h_v_l, h_v_l_s


class Decoding(nn.Module):
    def __init__(self, embedding_size, tensor_type, device):
        super().__init__()

        self.W_4 = nn.Parameter(tensor_init([embedding_size, 1], tensor_type, device))
        self.W_5 = nn.Parameter(tensor_init([embedding_size, 1], tensor_type, device))

        self.embedding_size = embedding_size

    def forward(
            self,
            actions: sparse.Tensor,  # [batch_size, node_cnt] or [node_cnt, batch_size]
            node_embedding: torch.Tensor,
            state_embedding: torch.Tensor,
            q_for_all=False
    ):
        if not q_for_all:
            batch_size = actions.shape[0]

            # Uppercase Z because many z's from different batches are integrated into one matrix
            # [batch_size, embedding_size] = [batch_size, node_cnt] * [node_cnt, embedding_size]
            Z_a = sparse.mm(actions, node_embedding)
            # [batch_size, embedding_size]
            Z_s = state_embedding

            # change the shape of z_a and z_s to multiply them row by row
            # [batch_size, embedding_size, 1]
            Z_a = torch.unsqueeze(Z_a, dim=2)
            # [batch_size, 1, embedding_size]
            Z_s = torch.unsqueeze(Z_s, dim=1)

            Q_s_a = torch.matmul(
                f.relu(
                    torch.reshape(
                        torch.matmul(
                            torch.matmul(Z_a, Z_s),
                            self.W_4
                        ),
                        [batch_size, self.embedding_size]
                    )
                ),
                self.W_5
            )
            return Q_s_a
        else:
            node_cnt = actions.shape[0]

            # We need Q value for all nodes as action
            # Thus get a state embedding for each node embedding according to node-batch relation

            # [node_cnt, embedding_size] = [node_cnt, batch_size] * [batch_size, embedding_size]
            Z_s = sparse.mm(actions, state_embedding)
            Z_a = node_embedding

            # change the shape of z_a and z_s to multiply them row by row
            # [node_cnt, embedding_size, 1]
            Z_a = torch.unsqueeze(Z_a, dim=2)
            # [node_cnt, 1, embedding_size]
            Z_s = torch.unsqueeze(Z_s, dim=1)

            Q_s_a_all = torch.matmul(
                f.relu(
                    torch.reshape(
                        torch.matmul(
                            torch.matmul(Z_a, Z_s),
                            self.W_4
                        ),
                        [node_cnt, self.embedding_size]
                    )
                ),
                self.W_5
            )
            return Q_s_a_all
