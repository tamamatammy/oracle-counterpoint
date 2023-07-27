# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 00:10:25 2020

@author: Tammy Yang

@description: Network construction code 

"""

import numpy as np
from scipy import linalg
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


def partial_corr(inv_cov):
    """form graphical model/partial correlation matrix (absent diagonals) given inverse covariance"""
    dinvsq = np.diag(1.0 / np.sqrt(np.diag(inv_cov)))
    gmodel = np.dot(dinvsq, np.dot(-inv_cov, dinvsq))
    gmodel = gmodel - np.diag(np.diag(gmodel))
    return gmodel


def sp_inv_cov_est(data):
    """Estimate sparse inverse covariance on dataset"""

    data = pd.DataFrame(data).drop(["date_time", "time"], axis=1).dropna(axis=0)
    # normalize data for relationship estimation
    data_scalar = StandardScaler().fit(data)
    data_standard = data_scalar.transform(data)

    model = GraphicalLassoCV(max_iter=2000)
    model.fit(data_standard)
    # cov_ = model.covariance_
    inv_cov_ = pd.DataFrame(model.precision_, columns=nodes_, index=nodes_)

    return inv_cov_, model


def create_nx_graph(nodes, A):
    """Create networkx graph from adjacency matrix A"""
    G = nx.Graph()
    mapping_ = {}
    for i in range(len(nodes)):
        G.add_node(i, label=nodes[i])
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])
        mapping_[i] = nodes[i]
    return G, mapping_
