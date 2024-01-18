import torch
from tqdm import tqdm
import random
import networkx as nx

from graph_utils import gen_graph, HXA, laplacian_matrix
from Q_network import QNetwork
from env import FinderEnvironment
from replay_buffer import ExperienceReplayBuffer
from graph_utils import accumulated_normalized_connectivity


def copy_params(dst_network: torch.nn.Module, src_network: torch.nn.Module):
    dst_network.load_state_dict(src_network.state_dict())


def prepare_validate_data(n_validate, num_min, num_max):
    print('\ngenerating validation graphs...')
    result_degree = 0
    result_betweeness = 0
    validate_set = []
    for i in tqdm(range(n_validate)):
        g = gen_graph(num_min, num_max)
        result_degree += HXA(g, 'HDA')[0]
        result_betweeness += HXA(g, 'HBA')[0]
        validate_set.append(g)
    print('Validation of HDA: %.6f' % (result_degree / n_validate))
    print('Validation of HBA: %.6f' % (result_betweeness / n_validate))
    return validate_set


def prepare_train_data(n_train, num_min, num_max):
    print('\ngenerating new training graphs...')
    train_set = []
    for i in tqdm(range(n_train)):
        train_set.append(gen_graph(num_min, num_max))
    return train_set


def prepare_batch(
        q_for_all: bool,
        tensor_type,
        device,
        env_list=None,
        s_t=None,
        a_t=None
):
    if q_for_all:
        if env_list is not None:
            # node_batch
            ones_graph = []
            for env in env_list:
                ones_graph.append(torch.ones([len(env.G.nodes), 1], dtype=tensor_type, device=device))
            node_batch = torch.block_diag(*tuple(ones_graph)).to_sparse_coo()
            # node_node
            adj_mat_list = []
            for env in env_list:
                adj_mat_list.append(
                    torch.tensor(nx.adjacency_matrix(env.G).todense(), dtype=tensor_type, device=device))
            node_node = torch.block_diag(*tuple(adj_mat_list)).to_sparse_coo()
            # batch_node
            batch_node = node_batch.T
            return node_batch, node_node, batch_node, True
        elif s_t is not None:
            ones_graph = []
            for s in s_t:
                ones_graph.append(torch.ones([len(s.shape[0]), 1], dtype=tensor_type, device=device))
            node_batch = torch.block_diag(*tuple(ones_graph)).to_sparse_coo()
            node_node = torch.block_diag(*tuple(s_t)).to_sparse_coo()
            batch_node = node_batch.T
            return node_batch, node_node, batch_node, True
        else:
            raise TypeError("prepare_batch expect either env_list or s_t when q_for_all == True")
    else:
        # node_node
        node_node = torch.block_diag(*tuple(s_t)).to_sparse_coo()
        # action and batch_node
        actions = torch.zeros([len(s_t), node_node.shape[0]], dtype=tensor_type, device=device)
        cnt = 0
        ones_graph = []
        for i in range(len(s_t)):
            actions[i][cnt + a_t[i]] = 1
            cnt += s_t[i].shape[0]
            # batch_node
            ones_graph.append(torch.ones([1, s_t[i].shape[0]], dtype=tensor_type, device=device))
        batch_node = torch.block_diag(*tuple(ones_graph)).to_sparse_coo()
        # laplacian
        l_lap = []
        for s in s_t:
            l_lap.append(laplacian_matrix(s))

        return actions, node_node, batch_node, False, True, torch.block_diag(*tuple(l_lap)).to_sparse_coo()


def play_game(
        n_experience,
        epsilon,
        n_step,
        online_network: QNetwork,
        env_list,
        replay_buffer: ExperienceReplayBuffer,
        train_set: list,
        tensor_type,
        device
):
    n_required = n_experience
    while n_required > 0:
        for env in env_list:
            if env.is_terminal_state():
                n_required = replay_buffer.from_state_sequence(
                    env.state_sequence,
                    env.action_sequence,
                    env.reward_sequence,
                    n_step,
                    n_required
                )
                env.load_graph(random.choice(train_set))

        if random.uniform(0, 1) > epsilon:
            # [num_nodes_in_all_envs, 1]
            Q_pred = online_network(
                *prepare_batch(env_list=env_list, q_for_all=True, tensor_type=tensor_type, device=device))
            cnt = 0
            for env in env_list:
                n_node = len(env.G.nodes)
                action = torch.argmax(Q_pred[cnt:cnt + n_node])
                cnt += n_node
                env.step(action)
        else:
            for env in env_list:
                action = random.randint(0, len(env.G.nodes) - 1)
                env.step(action)


def test(
        validate_set,
        online_network,
        n_step,
        n_validate,
        tensor_type,
        device,
        test_set=False
):
    robustness_sum = 0
    test_env = FinderEnvironment(n_step, tensor_type, device)
    result = []
    for i in range(n_validate):
        test_env.load_graph(validate_set[i])
        critical_nodes = []
        while not test_env.is_terminal_state():
            Q_pred = online_network(*prepare_batch(True, tensor_type, device, [test_env]))
            critical_nodes.append(test_env.deduction_step(torch.argmax(Q_pred)))
        sol = critical_nodes + list(set(validate_set[i].nodes) ^ set(critical_nodes))
        robustness_sum += accumulated_normalized_connectivity(sol, validate_set[i])
        if test_set:
            print(robustness_sum)
            result.append(robustness_sum)
            robustness_sum = 0

    if not test_set:
        return robustness_sum / n_validate
    else:
        return result
