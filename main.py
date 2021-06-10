import pandas as pd
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import seaborn as sns

import networkx as nx


def diffusion_centrality(g, t, q):
    '''
    g = adjacency matrix
    t = iterations
    q = probability
    '''
    arg1 = sum([(q * np.array(g)) ** i for i in range(1, t + 1)])
    return np.dot(arg1, np.ones(arg1.shape[0]))


def get_first_eigenval(g):
    e_val, e_vec = np.linalg.eig(g)
    q = 1 / e_val[0].real               # inverse of first eigenvalue of adjacency matrix is q
    return q


if __name__ == '__main__':

    panel = pd.read_stata('data/Stata Replication/data/panel.dta')
    max_ts = panel.groupby('village').t.max().to_dict()

    deg_centralities = {}
    deg_centralities_total = {}
    diff_centralities = {}
    diff_centralities_total = {}
    diff_centralities_q25 = {}
    diff_centralities_q25_total = {}
    diff_centralities_test = {}
    MF_empirical = {}
    for i in range(1, 78):
        try:
            mydata = genfromtxt(
                f'data/Data/1. Network Data/Adjacency Matrices/adj_allVillageRelationships_HH_vilno_{i}.csv',
                delimiter=',')
            has_leader = genfromtxt(
                f'/Users/polinab/gitworkspace/networks-2/networks-diffusion-centrality/data/Matlab Replication/India Networks/HHhasALeader{i}.csv',
                delimiter='\t')
            has_leader = np.array([int(x[1]) for x in has_leader])

            where_has_leader = np.where(np.array(has_leader) == 1)
            adjacency_all = mydata[1:, 1:]

            adjacency_leaders = np.array([x[where_has_leader] for x in adjacency_all[where_has_leader]])
            G_all = nx.Graph(adjacency_all)
            G_leaders = nx.Graph(adjacency_leaders)

            try:
                diff_centralities[i] = stats.mean(
                    [c for c in
                     diffusion_centrality(adjacency_leaders, int(max_ts[i]), get_first_eigenval(adjacency_leaders))])
                diff_centralities_q25[i] = stats.mean(
                    [c for c in diffusion_centrality(adjacency_leaders, int(max_ts[i]), 0.25)])
                diff_centralities_total[i] = stats.mean(
                    [c for c in diffusion_centrality(adjacency_all, int(max_ts[i]), get_first_eigenval(adjacency_all))])
                diff_centralities_q25_total[i] = stats.mean(
                    [c for c in diffusion_centrality(adjacency_all, int(max_ts[i]), 0.25)])
                diff_centralities_test[i] = \
                    panel.loc[(panel.village == i) & (panel.t == max_ts[i] - 1)].diffusion_centrality_leader.tolist()[0]
                MF_empirical[i] = \
                    panel.loc[(panel.village == i) & (panel.t == max_ts[i] - 1)].dynamicMF_empirical.tolist()[0]
                deg_centralities[i] = stats.mean([c for c in nx.degree_centrality(G_leaders).values()])
                deg_centralities_total[i] = stats.mean([c for c in nx.degree_centrality(G_all).values()])
            except:
                print(f'no records in panel data for village # {i}')

        except:
            print(f'no adjacency matrix data for village #{i}')

    df = pd.DataFrame([MF_empirical,
                       deg_centralities,
                       deg_centralities_total,
                       diff_centralities,
                       diff_centralities_q25,
                       diff_centralities_total,
                       diff_centralities_q25_total,
                       diff_centralities_test
                       ]).transpose()

    df.columns = ['MF_empirical',
                  'deg_centralities',
                  'deg_centralities_total',
                  'diff_centralities',
                  'diff_centralities_q25',
                  'diff_centralities_total',
                  'diff_centralities_q25_total',
                  'diff_centralities_test']

    #### TEST DATA ####

    # plotting given diffusion centralities from dataframe
    ax = sns.regplot(x="diff_centralities_test", y="MF_empirical", data=df) \
        .set_title('Diffusion Centrality vs MF, DC from Panel Data')
    plt.xlabel("Diffusion Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    #### PART 1 ####

    # plotting calculated degree centralities vs MF
    ax = sns.regplot(x="deg_centralities", y="MF_empirical", data=df) \
        .set_title('Degree Centrality of Leaders vs MF')
    plt.xlabel("Degree Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    # plotting my calculated diffusion centralities
    ax = sns.regplot(x="diff_centralities", y="MF_empirical", data=df) \
        .set_title('Diffusion Centrality vs MF, Q = 1st eigenvalue')
    plt.xlabel("Diffusion Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    #### PART 2 ####

    # plotting calculated degree centralities with q=0.25 vs MF
    ax = sns.regplot(x="diff_centralities_q25", y="MF_empirical", data=df) \
        .set_title('Diffusion Centrality vs MF, Q = 0.25')
    plt.xlabel("Diffusion Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    # plotting calculated degree centralities vs MF, normalized for all vilalges
    ax = sns.regplot(x="deg_centralities_total", y="MF_empirical", data=df) \
        .set_title('Degree Centrality vs MF, Normalized Across All HH')
    plt.xlabel("Degree Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    # plotting given diffusion centralities from dataframe
    ax = sns.regplot(x="diff_centralities_total", y="MF_empirical", data=df) \
        .set_title('Diffusion Centrality vs MF, Normalized Across All HH')
    plt.xlabel("Diffusion Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()

    # plotting calculated degree centralities with q=0.25 vs MF - over all villages
    ax = sns.regplot(x="diff_centralities_q25_total", y="MF_empirical", data=df) \
        .set_title('Diffusion Centrality vs MF, Normalized Across All HH, Q = 0.25')
    plt.xlabel("Diffusion Centrality")
    plt.ylabel("Microfinance participation rate")
    plt.show()
