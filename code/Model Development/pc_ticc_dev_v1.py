# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import sys
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, 'C:\\GitHub\\Project_Counterpoint\\TICC')
from TICC_solver import TICC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from plotnine import ggplot, geom_line, aes

###########################################################################

dir_base='C:/GitHub/Project_Counterpoint/'
dir_input='Data/Output/'
dir_output='Model Development/Output/'

###########################################################################

def prep_ticc_input(input_filename):
    df_0=pd.read_pickle(dir_input + input_filename+'.pk')
    df_1=df_0.drop(['time'], axis = 1).dropna(axis = 0)
    df_2=df_1.copy()
    df_array=df_2.to_numpy()#[0:10000,]
    df_scalar=StandardScaler().fit(df_array)
    df_standard = df_scalar.transform(df_array)
    df=df_standard
    np.savetxt(dir_base+'Model Development/'+input_filename+'.csv', df, delimiter=",")
    return df_0, df
    
input_raw, input_transformed = prep_ticc_input('btc_data_hourly_ema')

###########################################################################

lambda_param = 0.01

num_cluster = [2,3]#,4,5,6,7,8]
cluster_assignment=list()
cluster_MRFs=list()

for i in range(len(num_cluster)):
    
    ticc = TICC(window_size=1
            , number_of_clusters=num_cluster[i]
            , lambda_parameter = 0.01
            , beta=600
            , maxIters=100
            , threshold=2e-5
            , write_out_file=False
            , prefix_string="output_folder/"
            , num_proc=1)
    
    (cluster_assignment_i, cluster_MRFs_i) = ticc.fit(input_file='Model Development/df_input_array.csv')
    cluster_assignment.append(cluster_assignment_i)
    cluster_MRFs.append(cluster_MRFs_i)

input_raw['Cluster'] = cluster_assignment

input_raw.to_csv('input_raw.csv')
















def partial_corr(inv_cov):
    '''form graphical model/partial correlation matrix (absent diagonals) given inverse covariance'''
    dinvsq = np.diag(1./np.sqrt(np.diag(inv_cov)))
    gmodel = np.dot(dinvsq,
                    np.dot(-inv_cov, dinvsq))
    gmodel = gmodel - np.diag(np.diag(gmodel))
    gmodel[gmodel <=0 ] = 0
    return gmodel

x = partial_corr(cluster_MRFs[1])

np.count_nonzero(x_1)












x = cluster_MRFs[1]
df_input.columns

def partial_corr(inv_cov):
    '''form graphical model/partial correlation matrix (absent diagonals) given inverse covariance'''
    dinvsq = np.diag(1./np.sqrt(np.diag(inv_cov)))
    gmodel = np.dot(dinvsq,
                    np.dot(-inv_cov, dinvsq))
    gmodel = gmodel - np.diag(np.diag(gmodel))
    return gmodel


x_1=partial_corr(x)

def create_nx_graph(nodes, A, c_value):
    '''Create networkx graph from adjacency matrix A'''
    G = nx.Graph()
    mapping_ = {}
    for i in range(len(nodes)):
        G.add_node(i, label=nodes[i])
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if abs(A[i,j]) > c_value:
                G.add_edge(i,j,weight=abs(A[i,j]))
        mapping_[i] = nodes[i]
    return G,mapping_

G, mapping = create_nx_graph(list(df_input.columns), x_1, 1e-10)

nx.write_gexf(G, "test.gexf")

G = nx.petersen_graph()

plt.subplot(121)

nx.draw(G, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_shell(G, with_labels=True)#, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')


###########################################################################

# pacakge sample code#
fname = "TICC\\example_data.txt"
ticc = TICC(window_size=1, number_of_clusters=8, lambda_parameter=11e-2, beta=600, maxIters=1, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1)
(cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

print(cluster_assignment)
np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
