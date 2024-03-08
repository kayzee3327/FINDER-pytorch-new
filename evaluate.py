import torch
import networkx as nx
from Q_network import QNetwork
from train_utils import prepare_batch

model_folder = './models/'
state_l = []
for i in range(300, 9901, 300):
    model_name = model_folder + 'nrange_30_50_episode_' + str(i) + '.pt'
    state_l.append(torch.load(model_name, map_location=torch.device('cpu')))

graph_folder = './data/'
graph_names = [
    'Crime.txt',
    'Digg.txt',
    'Enron.txt',
    'Epinions.txt',
    'HI-II-14.txt'
]

graph_l = []
for name in graph_names:
    graph_l.append(nx.read_edgelist(graph_folder + name))

GAMMA = 1  # decay rate of past observations
UPDATE_TIME = 1000
EMBEDDING_SIZE = 64
# MAX_ITERATION = 500000
LEARNING_RATE = 0.0001  # dai
MEMORY_SIZE = 500000
Alpha = 0.001  # weight of reconstruction loss

# hyperparameters for priority
# epsilon = 0.0000001  # small amount to avoid zero priority
# alpha = 0.6  # [0~1] convert the importance of TD error to priority
# beta = 0.4  # importance-sampling, from initial value increasing to 1
# beta_increment_per_sampling = 0.001
# TD_err_upper = 1.  # clipped abs error

# other network para
N_STEP = 5
NUM_MIN = 30
NUM_MAX = 50
# REG_HIDDEN = 32
BATCH_SIZE = 64
# initialization_stddev = 0.01  # 权重初始化的方差
n_valid = 200
# aux_dim = 4
num_env = 1
# inf = 2147483647 / 2

# embedding method
K = 3
# aggregatorID = 0  # 0:sum; 1:mean; 2:GCN
# embeddingMethod = 1  # 0:structure2vec; 1:graphsage

# my parameter
INPUT_SIZE = 2  # size of input feature, c in Algorithm S2
N_train = 1000
N_episode = 10000
T = 10
C = UPDATE_TIME
UPDATE_GRAPHS = 100
tensor_type = torch.float32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test(model, graph):
    return model(*prepare_batch(q_for_all=True,
                                tensor_type=tensor_type,
                                device=device,
                                s_t=[torch.tensor(nx.adjacency_matrix(graph).todense(), device=device, dtype=tensor_type)]))


q = QNetwork(INPUT_SIZE, EMBEDDING_SIZE, K, tensor_type, device)
q.load_state_dict(state_l[0])
test(q, graph_l[0])
