import random

import torch
import torch.nn.functional as F
import time

from Q_network import QNetwork
from train_utils import copy_params, prepare_validate_data, prepare_train_data, play_game, prepare_batch, test
from replay_buffer import ExperienceReplayBuffer
from env import FinderEnvironment

GAMMA = 1  # decay rate of past observations
UPDATE_TIME = 1000
# EMBEDDING_SIZE = 64
# MAX_ITERATION = 500000
LEARNING_RATE = 0.0001  # dai
# MEMORY_SIZE = 500000
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
# K = 3
# aggregatorID = 0  # 0:sum; 1:mean; 2:GCN
# embeddingMethod = 1  # 0:structure2vec; 1:graphsage

# my parameter
N_episode = 10000
T = 50
C = UPDATE_TIME
UPDATE_GRAPHS = 100
tensor_type = torch.float32

n_step_replay_buffer = ExperienceReplayBuffer()
online_Q = QNetwork()
target_Q = QNetwork()
optimizer = torch.optim.Adam(online_Q.parameters(), lr=LEARNING_RATE)

copy_params(target_Q, online_Q)

# below graph set operation is not shown in Algorithm S3
validate_set = prepare_validate_data(n_valid)
train_set = prepare_train_data()

# In author's code, the idea of episode in Algorithm S3 is discarded.
# Instead, author obtains 1000 random-action 4-tuples by PlayGame on graphs in train_set in advance,
# in order to ensure mini-batch training with enough data.
# Moreover, in author's code, 10 experiences are added to replay buffer every 10 trainings.
# Validation for every 300 trainings and Update of target network for every C trainings.

# Here I adjusted some implementation ideas of author's code:
# 1. use episode to cater to change of epsilon, the probability of random action
# 2. T trainings every episode
# 3. validate every 300 episodes just as the original work does
# 4. add 10 experiences every 10 trainings (idea from code but not in paper)
# 5. Update target network every C(UPDATE_TIME) trainings
# 6. Get new training graphs every UPDATE_GRAPHS episodes (idea from code but not in paper)

env_list = []
for i in range(num_env):
    env = FinderEnvironment(N_STEP)
    env.load_graph(random.choice(train_set))
    env_list.append(env)
play_game(1000, 1, N_STEP, online_Q, env_list, n_step_replay_buffer, train_set)

total_training_cnt = 0
epsilon_start = 1.0
epsilon_end = 0.05
time_300_episode = time.time()
for episode in range(1, N_episode+1):
    epsilon = epsilon_end + max(0., (epsilon_start - epsilon_end) * (N_episode - episode) / N_episode)
    if episode % 300 == 0:

        print('300 episodes total time: %.2fs\n' % (time.time() - time_300_episode))
        t_start = time.time()
        robustness = test(validate_set, online_Q, N_STEP, n_valid)
        t_end = time.time()
        print('iter %d, eps %.4f, average size of vc:%.6f' % (episode, epsilon, robustness))
        print('testing 200 graphs time: %.2fs' % (t_end - t_start))

        model_path = '%s/nrange_%d_%d_iter_%d.pt' % (
            "/Users/wangkaize/Desktop/Flight-Delay/FINDER/FINDER-pytorch-new/model",
            NUM_MIN, NUM_MAX, episode
        )

    if episode % UPDATE_GRAPHS == 0:
        train_set = prepare_train_data()
    for t in range(T):
        optimizer.zero_grad()
        s_t, a_t, r, s_t_n, terminal = n_step_replay_buffer.sample(BATCH_SIZE)
        # calculate y_j, as in Q_true
        r = torch.tensor(r, dtype=tensor_type)
        Q_s_t_n_chosen = torch.zeros(len(r), dtype=tensor_type)
        if False in terminal:
            Q_s_t_n = target_Q(prepare_batch(True, s_t=s_t_n))
            Q_s_t_n_max = torch.zeros(len(s_t_n), dtype=tensor_type)
            cnt = 0
            for i in range(len(s_t_n)):
                Q_s_t_n_max[i] = torch.max(Q_s_t_n[cnt:cnt + s_t_n[i].shape[0]])
            Q_s_t_n_chosen = torch.where(torch.tensor(terminal), 0, 1) * Q_s_t_n_max
        Q_true = r + GAMMA * Q_s_t_n_chosen
        # calculate Q_pred and loss
        Q_pred, loss_graph_recons = online_Q(*prepare_batch(q_for_all=False, s_t=s_t, a_t=a_t))
        Q_pred = torch.reshape(Q_pred, [Q_true.shape[0]])
        loss_rl = F.mse_loss(Q_pred, Q_true)
        loss = loss_rl + Alpha * loss_graph_recons

        loss.backward()
        optimizer.step()
        total_training_cnt += 1

        if total_training_cnt % 10 == 0:
            play_game(10, epsilon, N_STEP, online_Q, env_list, n_step_replay_buffer, train_set)
        if total_training_cnt % UPDATE_TIME == 0:
            copy_params(target_Q, online_Q)





