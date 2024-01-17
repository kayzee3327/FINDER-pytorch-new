import networkx as nx
import torch

from graph_utils import pairwise_connectivity


class FinderEnvironment:
    def __init__(self, n_step, tensor_type, device, measurement="pairwise connectivity"):
        self.G = None
        self.state_sequence = []
        self.action_sequence = []
        self.reward_sequence = []
        if measurement == "pairwise connectivity":
            self.connectivity = pairwise_connectivity
        else:
            raise Exception('undefined measurement')

        self.initial_connectivity = 0
        self.N = 0
        self.tensor_type = tensor_type
        self.device = device

    def load_graph(self, graph: nx.Graph()):
        self.G = graph.copy()
        self.state_sequence = [torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=self.tensor_type, device=self.device)]  # s_0
        self.action_sequence = []
        self.reward_sequence = []
        self.initial_connectivity = self.connectivity(graph)
        self.N = len(graph.nodes)

    def step(self, action):
        self.G.remove_node(list(self.G.nodes)[action])
        self.state_sequence.append(torch.tensor(nx.adjacency_matrix(self.G).todense(), dtype=self.tensor_type, device=self.device))  # s_t+1
        self.action_sequence.append(action)  # a_t
        self.reward_sequence.append(self.get_reward())  # r_t

    def deduction_step(self, action):
        node_name = list(self.G.nodes)[action]
        self.G.remove_node(node_name)
        return node_name

    def get_reward(self):
        return -1 * (self.connectivity(self.G) / (self.N * self.initial_connectivity))

    def is_terminal_state(self):
        # catering to your need
        # below is an example
        return self.connectivity(self.G) < self.initial_connectivity * 0.1

