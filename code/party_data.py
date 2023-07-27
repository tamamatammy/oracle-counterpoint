# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:33:06 2021

@author: Tammy Yang

party_data module for data manipulation 

"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt

class party_data:
    def __init__(self,df):
        self.df=df.copy()
        
    def by_type(self,party_key,excl=True):
        "partition data by data type defined by column name"
        col_party=[]
        for pk in party_key:
            col_party+=[x for x in self.df.columns if pk in x]
        
        if excl==True:
            self.df=self.df.drop(col_party, axis=1)
        else:
            self.df=self.df[col_party]
        return self
    
    def by_date(self,party_key,gt=True):
        "partition data by dates"
        
        self.df.index=pd.to_datetime(self.df.index, format='%Y-%m-%d')
        if gt==True:
            self.df=self.df[self.df.index>=pd.to_datetime(party_key, format='%Y-%m-%d').tz_localize("UTC")]
        else:
            self.df=self.df[self.df.index<pd.to_datetime(party_key, format='%Y-%m-%d').tz_localize("UTC")]
        return self
    
    def to_lag(self, col_list, lag):
        "lag selected columns"
        self.df.sort_index(inplace=True)
        for col in col_list:
            self.df[col]=(self.df[col]-self.df[col].shift(lag))/self.df[col].shift(lag)
        return self
    
    def stable_var(self, col_list):
        "lag selected columns"
        for col in col_list:
            col_box_cox,_=stats.boxcox(self.df[col])
            self.df[col]=col_box_cox
        return self
    
    def de_trend(self, col_list,lag):
        "lag selected columns"
        for col in col_list:
            self.df[col]=self.df[col]-self.df[col].shift(lag)
        return self
            
    
    def to_smooth(self, col_list, alpha=1, span=1, min_periods=100):
        "smooth selected columns"
        if alpha<1:
            for col in col_list:
                self.df[col]=self.df[col].ewm(alpha=alpha, ignore_na=True, min_periods=min_periods, adjust=False).mean()
        elif span>1:
            for col in col_list:
                self.df[col]=self.df[col].ewm(span=span, ignore_na=True, min_periods=min_periods, adjust=False).mean()

        return self
    
def load_pickle_streams(filename):
    data = []
    with open(filename, "rb") as reader:
        try:
            while True:
                data.append(pickle.load(reader))
        except EOFError:
            pass
        reader.close()
    return data

def plot_ts(df_0):
    for col in df_0.columns:
        plt.figure(figsize=(16, 3))
        plt.plot(df_0[col])
        plt.title(label="Time Series: "+col, fontsize=10)
        plt.xlim(min(df_0.index), max(df_0.index))
        plt.show()
        
def split_test_train(df_0, col_response, split_date="2019-06-01", test_size=0.2,random_switch=True):
    
    col_predictors=[c for c in df_0.columns if c != col_response]
    #response = response
    if random_switch==False:
        date_validate=split_date
        df_train=party_data(df_0).by_date(date_validate, gt=False).df
        df_validate=party_data(df_0).by_date(date_validate, gt=True).df
        
        x_train=df_train[col_predictors]
        x_test=df_validate[col_predictors]

        y_train=np.array(df_train[col_response])
        y_test=np.array(df_validate[col_response])
    else:
        x=df_0[col_predictors]
        y=df_0[col_response]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=123)
    
    return x_train, x_test, y_train, y_test

# def plot_act_pred(act_test, pred_test, act_train, pred_train, incl_train=True, title="",save=True,filename=""):
    
#     plt.figure(figsize=(8,8))
#     plt.scatter(act_test, pred_test, color="red", s=0.01)
#     if incl_train==True:
#         plt.scatter(act_train, pred_train, color="blue", s=0.01)
#         #plt.plot(pred_train, pred_train, color = 'black', linewidth = 0.5)
#         plt.xlim(min(min(pred_test), min(act_test), min(act_train), min(pred_train)), max(max(pred_test), max(act_test), max(act_train), max(pred_train)))
#         plt.ylim(min(min(pred_test), min(act_test), min(act_train), min(pred_train)), max(max(pred_test), max(act_test), max(act_train), max(pred_train)))
#     else:
#         plt.xlim(min(min(pred_test), min(act_test)), max(max(pred_test), max(act_test)))
#         plt.ylim(min(min(pred_test), min(act_test)), max(max(pred_test), max(act_test)))
    
#     plt.xlabel("Actual", fontsize=20)
#     plt.ylabel("Prediction", fontsize=20)
#     plt.title(title, fontsize=20)
#     if save==True:
#         plt.savefig(filename+".png")
#     plt.show()

def plot_act_pred(act_test, pred_test, title="",save=True,filename=""):
    
    plt.figure(figsize=(8,8))
    plt.scatter(act_test, pred_test, color="red", s=0.01)
   
    plt.xlim(min(min(pred_test), min(act_test)), max(max(pred_test), max(act_test)))
    plt.ylim(min(min(pred_test), min(act_test)), max(max(pred_test), max(act_test)))
    
    plt.xlabel("Actual", fontsize=20)
    plt.ylabel("Prediction", fontsize=20)
    plt.title(title, fontsize=20)
    if save==True:
        plt.savefig(filename+".png")
    plt.show()
# def var_imp(df_0, response):
    
#     df_1=df_0.copy()
#     df_1["random"]=np.random.normal(0,1,len(df_1))
    
#     train_col=[c for c in df_1.columns if c != response]
#     y_train=np.array(df_1[response])
#     x_train=df_1[train_col]
        
#     rf=RandomForestRegressor(n_estimators=500,max_samples=0.66,max_features=6, n_jobs=-1)
#     rf.fit(x_train, y_train)
    
#     var_import_obj=permutation_importance(rf, x_train, y_train, n_repeats=5, random_state=0)
#     var_imp=pd.DataFrame()
#     var_imp["Variable"]=x_train.columns
#     var_imp["Importances_Mean"]=var_import_obj.importances_mean
    
#     return var_imp


# def load_pickle_streams(filename):
#     data = []
#     with open(filename, "rb") as reader:
#         try:
#             while True:
#                 data.append(pickle.load(reader))
#         except EOFError:
#             pass
#         reader.close()
#     return data