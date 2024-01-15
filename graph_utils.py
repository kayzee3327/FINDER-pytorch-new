import networkx as nx
import numpy as np
import torch


def pairwise_connectivity(graph: nx.Graph):
    ans = 0
    components = nx.connected_components(graph)
    for component in components:
        n = len(component)
        ans += n * (n - 1) / 2
    return ans


def accumulated_normalized_connectivity(sequence, graph, weights=None, measurement='pairwise connectivity'):
    g = graph.copy()
    anc = 0
    if measurement == 'pairwise connectivity':
        connectivity = pairwise_connectivity
    else:
        raise Exception('undefined measurement')

    N = len(g.nodes)
    initial_connectivity = connectivity(g)
    for i in sequence:
        g.remove_node(i)
        if weights is None:  # node-unweighted
            anc += connectivity(g)
        else:  # node-unweighted
            anc += connectivity(g) * weights[i]
    return anc / (N * initial_connectivity)


def HXA(graph, method):
    sol = []
    g = graph.copy()
    while nx.number_of_edges(g) > 0:
        if method == 'HDA':
            c = nx.degree_centrality(g)
        elif method == 'HBA':
            c = nx.betweenness_centrality(g)
        elif method == 'HCA':
            c = nx.closeness_centrality(g)
        elif method == 'HPRA':
            c = nx.pagerank(g)
        else:
            raise Exception('undefined method')
        vals = list(c.values())
        max_node_id = list(g.nodes)[np.argmax(vals)]
        sol.append(max_node_id)
        g.remove_node(max_node_id)
    anc = accumulated_normalized_connectivity(sol, graph)
    return anc, sol


def gen_graph(min_node_num, max_node_num):
    node_num = np.random.randint(max_node_num-min_node_num+1) + min_node_num
    return nx.barabasi_albert_graph(node_num, 4)


def laplacian_matrix(adj_mat: torch.Tensor):
    D = torch.diag(torch.sum(adj_mat, dim=1))
    return D - adj_mat

