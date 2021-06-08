import pandas as pd
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

import networkx as nx


def diffusion_centrality(g, t, q):
    '''
    g = adjacency matrix
    t = iterations
    q = probability
    '''
    arg1 = sum([(q * np.array(g)) ** i for i in range(1, t + 1)])
    return np.dot(arg1, np.ones(arg1.shape[0]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    panel = pd.read_stata('data/Stata Replication/data/panel.dta')
    max_ts = panel.groupby('village').t.max().to_dict()

    deg_centralities = {}
    diff_centralities = {}
    diff_centralities_leader = {}
    MF_empirical = {}
    for i in range(1, 78):
        try:
            mydata = genfromtxt(
                f'data/Data/1. Network Data/Adjacency Matrices/adj_allVillageRelationships_HH_vilno_{i}.csv',
                delimiter=',')
        except:
            print(f'no adjacency matrix data for village #{i}')

        adjacency = mydata[1:, 1:]
        G = nx.Graph(adjacency)
        e_val, e_vec = np.linalg.eig(adjacency)
        q = 1 / e_val[0].real  # inverse of first eigenvalue of adjacency matrix is q
        deg_centralities[i] = stats.mean([c for c in nx.degree_centrality(G).values()])
        try:
            diff_centralities[i] = stats.mean([c for c in diffusion_centrality(adjacency, int(max_ts[i]), q)])
            diff_centralities_leader[i] = panel.loc[(panel.village == i) & (panel.t == max_ts[i] - 1)].diffusion_centrality_leader.tolist()[0]
            MF_empirical[i] = panel.loc[(panel.village == i) & (panel.t == max_ts[i] - 1)].dynamicMF_empirical.tolist()[0]
        except:
            print(f'no records in panel data for village # {i}')


    print(deg_centralities)
    print(diff_centralities)
    print(MF_empirical)

    plt.scatter([x for x in diff_centralities.values()], [x for x in MF_empirical.values()])
    plt.show()

    plt.scatter([x for x in diff_centralities_leader.values()], [x for x in MF_empirical.values()])
    plt.show()