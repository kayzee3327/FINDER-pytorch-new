import torch
from tqdm import tqdm
import random
import networkx as nx

from graph_utils import gen_graph, HXA, laplacian_matrix
from Q_network import QNetwork
from env import FinderEnvironment
from replay_buffer import ExperienceReplayBuffer
from graph_utils import accumulated_normalized_connectivity

# GAMMA = 1  # decay rate of past observations
# UPDATE_TIME = 1000
# EMBEDDING_SIZE = 64
# MAX_ITERATION = 500000
# LEARNING_RATE = 0.0001  # dai
# MEMORY_SIZE = 500000
# Alpha = 0.001  # weight of reconstruction loss

# hyperparameters for priority
# epsilon = 0.0000001  # small amount to avoid zero priority
# alpha = 0.6  # [0~1] convert the importance of TD error to priority
# beta = 0.4  # importance-sampling, from initial value increasing to 1
# beta_increment_per_sampling = 0.001
# TD_err_upper = 1.  # clipped abs error

# other network para
# N_STEP = 5
NUM_MIN = 30
NUM_MAX = 50
# REG_HIDDEN = 32
# BATCH_SIZE = 64
# initialization_stddev = 0.01  # 权重初始化的方差
n_valid = 200
# aux_dim = 4
# num_env = 1
# inf = 2147483647 / 2

# embedding method
# K = 3
# aggregatorID = 0  # 0:sum; 1:mean; 2:GCN
# embeddingMethod = 1  # 0:structure2vec; 1:graphsage

# my parameters
N_train = 1000


def copy_params(dst_network: torch.nn.Module, src_network: torch.nn.Module):
    dst_network.load_state_dict(src_network.state_dict())


def prepare_validate_data(n_validate=n_valid):
    print('\ngenerating validation graphs...')
    result_degree = 0
    result_betweeness = 0
    validate_set = []
    for i in tqdm(range(n_validate)):
        g = gen_graph(NUM_MIN, NUM_MAX)
        result_degree += HXA(g, 'HDA')[0]
        result_betweeness += HXA(g, 'HBA')[0]
        validate_set.append(g)
    print('Validation of HDA: %.6f' % (result_degree / n_valid))
    print('Validation of HBA: %.6f' % (result_betweeness / n_valid))
    return validate_set


def prepare_train_data(n_train=N_train):
    print('\ngenerating new training graphs...')
    train_set = []
    for i in tqdm(range(n_train)):
        train_set.append(gen_graph(NUM_MIN, NUM_MAX))
    return train_set


def prepare_batch(
        q_for_all: bool,
        env_list=None,
        s_t=None,
        a_t=None,
        tensor_type=torch.float32
):
    if q_for_all:
        if env_list is not None:
            # node_batch
            ones_graph = []
            for env in env_list:
                ones_graph.append(torch.ones([len(env.G.nodes), 1], dtype=tensor_type))
            node_batch = torch.block_diag(*tuple(ones_graph)).to_sparse_coo()
            # node_node
            adj_mat_list = []
            for env in env_list:
                adj_mat_list.append(torch.tensor(nx.adjacency_matrix(env.G).todense(), dtype=tensor_type))
            node_node = torch.block_diag(*tuple(adj_mat_list)).to_sparse_coo()
            # batch_node
            batch_node = node_batch.T
            return node_batch, node_node, batch_node, True
        elif s_t is not None:
            ones_graph = []
            for s in s_t:
                ones_graph.append(torch.ones([len(s.shape[0]), 1], dtype=tensor_type))
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
        actions = torch.zeros([len(s_t), node_node.shape[0]], dtype=tensor_type)
        cnt = 0
        ones_graph = []
        for i in range(len(s_t)):
            actions[i][cnt + a_t[i]] = 1
            cnt += s_t[i].shape[0]
            # batch_node
            ones_graph.append(torch.ones([1, s_t[i].shape[0]], dtype=tensor_type))
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
        train_set: list
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
            Q_pred = online_network(*prepare_batch(env_list=env_list, q_for_all=True))
            cnt = 0
            for env in env_list:
                n_node = len(env.G.nodes)
                action = torch.argmax(Q_pred[cnt:cnt + n_node])
                cnt += n_node
                env.step(action)
        else:
            for env in env_list:
                action = random.randint(0, len(env.G.nodes)-1)
                env.step(action)


def test(
        validate_set,
        online_network,
        n_step,

        n_validate=n_valid
):
    robustness_sum = 0
    test_env = FinderEnvironment(n_step)

    for i in range(n_validate):
        test_env.load_graph(validate_set[i])
        critical_nodes = []
        while not test_env.is_terminal_state():
            Q_pred = online_network(*prepare_batch(True, [test_env]))
            critical_nodes.append(test_env.deduction_step(torch.argmax(Q_pred)))
        sol = critical_nodes + list(set(validate_set[i].nodes) ^ set(critical_nodes))
        robustness_sum += accumulated_normalized_connectivity(sol, validate_set[i])

    return robustness_sum / n_validate


