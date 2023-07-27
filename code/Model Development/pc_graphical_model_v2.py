# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:38:16 2021

@author: yang8
"""

import pickle 
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import time

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pyeemd import ceemdan

import os
from settings import DEV_PATH, DATA_PATH,PROJECT_ROOT
os.chdir(DEV_PATH)
from party_data import party_data, plot_act_pred,split_test_train,plot_ts
import mutual_information as mi
os.chdir(PROJECT_ROOT)
#from celo_datasets_maker import process_transfers_ts,process_blocks_ts

#######################################################################################
# calculate sparse inverse covariance#
def sp_inv_cov_est(data,n_tols=1e-4,n_cv=5, n_alphas=4, n_refines=4, n_iters=1000):
    '''Estimate sparse inverse covariance on dataset'''
    random.seed(123)
    data_scalar = StandardScaler().fit(data)
    data_standard = data_scalar.transform(data)
    
    nodes_ = data.columns
    model_ = GraphicalLassoCV(alphas=n_alphas,tol=n_tols,cv=n_cv, n_refinements=n_refines, max_iter=n_iters)
    model_.fit(data_standard)
    #cov_ = model_.covariance_
    inv_cov_ = pd.DataFrame(model_.precision_, columns = nodes_, index = nodes_)
    
    return inv_cov_, model_#,cov_, inv_cov_, nodes_

# caluclate partial correlation
def partial_corr(inv_cov):
    '''form graphical model/partial correlation matrix (absent diagonals) given inverse covariance'''
    dinvsq = np.diag(1./np.sqrt(np.diag(inv_cov)))
    partial_corr = np.dot(dinvsq, np.dot(-inv_cov, dinvsq))
    partial_corr = partial_corr - np.diag(np.diag(partial_corr))    
    partial_corr_ = pd.DataFrame(partial_corr, columns = inv_cov.columns, index =  inv_cov.columns)    
    return partial_corr_



def filter_network(graph,response, excl_off_off=False, excl_on_on=False, node_threshold=0):
    g=graph.copy() 
    node_on_chain=response
    node_off_chain=[c for c in df_dev.columns if c != response]  
    for n1, n2, w in graph.edges(data=True):
        if abs(w["weight"])<=node_threshold:
            g.remove_edge(n1, n2)   
        if excl_off_off==True:
            if (n1 in node_off_chain) and (n2 in node_off_chain):
                g.remove_edge(n1,n2)
        if excl_on_on==True:
            if (n1 in node_on_chain) and (n2 in node_on_chain):
                g.remove_edge(n1,n2)  
    return g
 

def plot_rolling_pred(df_0,title,save,filename):
    plot_act_pred(df_0.Actual
                  ,df_0.Pred
                  ,df_0.Actual
                  ,df_0.Pred
                  ,incl_train=True
                  ,title=title
                  ,save=save
                  ,filename=filename
                  )


#######################################################################################
# load input
dir_input="Data/final/"
df_econ_0=pd.read_pickle(dir_input+'df_economic.pk')
df_final_0=pd.read_pickle(dir_input+'df_final.pk')
df_final_1=party_data(df_final_0).by_type(["pb","pt","weighted"],excl=False).df # only include transformed fields


# consolidate dev dataset 
df_all_0=pd.concat([df_final_1, df_econ_0], axis=1, join="outer")

col_excl=["open","high","low","volume"
          ,"usdc","USDC","Dai","dai","SAI","sai","usdt","ethAmount"
          ]

col_incl_df=pd.read_csv("Data/pc_lookup.csv")
col_incl=col_incl_df.loc[col_incl_df.incl=="y"]["Factor"]


df_all_1=party_data(df_all_0).by_type(col_excl).by_date("2016-07-01").by_date("2020-09-01",gt=False).df

df_all_2=df_all_1[col_incl]

df_all=df_all_2
col_all=df_all.columns
del df_all_0, df_all_1, df_all_2,col_excl,col_incl_df,col_incl,df_econ_0,df_final_0,df_final_1

# check duplicated columns #
col_all=list(df_all.columns)
col_all_rep=[c for c in col_all if col_all.count(c)>1]


# backfill nan and inf with prev values
df_dev=df_all.copy()
df_dev.replace(np.inf,np.NaN, inplace=True)
df_dev.fillna(method="ffill",inplace=True)

#######################################################################################
# graphical modeel #
inv_cov,_=sp_inv_cov_est(df_dev,n_tols=1e-4,n_cv=5, n_alphas=4, n_refines=4, n_iters=1000)
par_corr=partial_corr(inv_cov)

#######################################################################################
# rolling ensemble #

def rolling_forest(df,response, c=1):

    n=len(df)-1
   #c=1#2*12*24*30
    response=response
    train_col=[c for c in df.columns if c != response]
    df_pred_list=[]
    
    start=time.time()
    while n>500:#(c*200):#len(df)-1:#20*10*24:
    
        print(n)
    
        df_test=df.iloc[n-c:n]
        y_test=np.array(df_test[response])
        x_test=df_test[train_col]
    
        df_train=df.iloc[0:n-c]
        y_train=np.array(df_train[response])
        x_train=df_train[train_col]
        
        rf=RandomForestRegressor(n_estimators=10,max_samples=0.66,max_features=20, n_jobs=-1)
        rf.fit(x_train, y_train)
        y_pred=rf.predict(x_test)
        
    
        df_pred_n=pd.DataFrame()
        df_pred_n["Pred"]=y_pred
        df_pred_n["Actual"]=y_test
        df_pred_n.index=x_test.index
        #df_pred_n=pd.DataFrame([[y_test.item(), y_pred.item()]], columns=["Actual","Pred"], index=df_test.index)
        df_pred_list.append(df_pred_n)
        
        n=n-c

    end = time.time()
    runtime_secondes=end - start
    print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(runtime_secondes)))

    df_pred_rolling_forest=pd.concat(df_pred_list)
    df_pred_rolling_forest.sort_index(inplace=True)
    
    return df_pred_rolling_forest

df_pred_rolling_rf=rolling_forest(df_dev,"eth_close", c=24)










