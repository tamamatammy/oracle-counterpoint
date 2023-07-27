import pandas as pd
import numpy as np
import random
import itertools as iter
from datetime import tzinfo
import pickle as pk
import time
from joblib import dump, load


from scipy import linalg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,RandomizedSearchCV
from sklearn.inspection import permutation_importance
import xgboost as xgb
import scipy.stats as stats


import matplotlib.pyplot as plt
import matplotlib.axes as ax

###############################################################################
dir_input = 'Data/Output/'

dir_output = 'Model Development/Output/'

df_final = pd.read_pickle(dir_input + 'df_final.pk')

node_off_chain=["eth_close"
                ,"eth_spread_proxy"
                ,"eth_volumn"
                ,'btc_close'
                ,'btc_spread_proxy'
                ,'btc_volumn'
                ]

col_growth=["btc_difficulty"
            ,"btc_block_size"
            ,"btc_n_to_address"
            ,"btc_n_fr_address"
            ,"btc_close"
            ,"eth_n_to_address"
            ,"eth_n_fr_address"
            ,"eth_n_tx_per_block"
            #,"eth_ethersupply_sum"
            ,"eth_blocksize"
            ,"eth_gasused"
            ,"eth_gaslimit"
            ,"eth_difficulty"
            ,"eth_close"]



###############################################################################

class party_data:
    def __init__(self,df):
        self.df=df.copy()
        
    def by_type(self,party_key,excl=True):
        "partition data by data type defined by column name"
        col_party=[x for x in self.df.columns if party_key in x]
        
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
            
    
    def to_smooth(self, col_smooth, alpha):
        "smooth selected columns"
        for col in col_smooth:
            self.df[col]=self.df[col].ewm(alpha=alpha, ignore_na = True).mean()
        return self
    

def split_test_train(df_0,response, split_date="2019-06-01", test_size=0.2,random_switch=True):
    
    node_on_chain=[n for n in df_0.columns if n not in node_off_chain]

    response = response

    if random_switch==False:
        date_validate=split_date
        df_train=party_data(df_0).by_date(date_validate, gt=False).df
        df_validate=party_data(df_0).by_date(date_validate, gt=True).df
        
        x_train=df_train[node_on_chain]
        x_test=df_validate[node_on_chain]

        y_train=np.array(df_train[response])
        y_test=np.array(df_validate[response])
    else:
        x=df_0[node_on_chain]
        y=df_0[response]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=123)
    
    return x_train, x_test, y_train, y_test

   
def plot_ts(df_0):
    for col in df_0.columns:
        plt.figure(figsize=(16, 3))
        plt.plot(df_0[col])
        plt.title(label="Time Series: "+col, fontsize=10)
        plt.show()
        
      
def plot_act_pred(act_test,pred_test, act_train, pred_train):
    plt.figure(figsize=(8,8))
    plt.scatter(act_test, pred_test, color="blue", s=1)
    plt.scatter(act_train, pred_train, color="red", s=1)
    plt.plot(act_train, act_train, color = 'black', linewidth = 2)
    plt.xlim(min(min(pred_train), min(act_train), min(act_train), min(pred_train)), max(max(pred_train), max(act_train), min(act_train), min(pred_train)))
    plt.ylim(min(min(pred_train), min(act_train)), max(max(pred_train), max(act_train)))
    plt.xlabel("Actual", fontsize=20)
    plt.ylabel("Prediction", fontsize=20)
    plt.title("Actual vs Prediction ", fontsize=20)
    plt.show()

###################################################################################


df_eth=party_data(df_final.drop(["eth_spread_proxy", "eth_volumn"],axis=1)).by_type(party_key="eth", excl=False).by_date(party_key="2020-01-01", gt=False).by_date(party_key="2016-07-01", gt=True).df
df_eth.dropna(inplace=True)

df_eth_transform=party_data(df_eth).stable_var(df_eth.columns).de_trend(df_eth.columns, 1).df

df_eth_lag=party_data(df_eth).to_lag(col_list=df_eth.columns, lag=1).df
df_eth_lag.dropna(inplace=True)

#plot_ts(df_eth_transform)
#plot_ts(df_eth)
df=party_data(df_btc_eth_econ).by_date(party_key="2020-01-01", gt=False).by_date(party_key="2016-07-01", gt=True).df

##################################################################################
# rolling ensemble #
n=len(df)-1
c=2*12*24*30
response="eth_close"
train_col=[c for c in df.columns if c != response]
#rf=RandomForestRegressor(n_estimators=10,n_jobs=-1)
df_pred_list=[]

start=time.time()
while n>len(df)-20*10*24:
    
    print(n)
    
    df_test=df.iloc[n-10*24:n]
    y_test=np.array(df_test[response])
    x_test=df_test[train_col]
    
    df_train=df.iloc[n-10*24-c:n-10*24]
    y_train=np.array(df_train[response])
    x_train=df_train[train_col]
    
    rf=RandomForestRegressor(n_estimators=100,n_jobs=-1)
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    
    
    df_pred_n=pd.DataFrame()
    df_pred_n["Pred"]=y_pred
    df_pred_n["Actual"]=y_test
    df_pred_n.index=x_test.index
    #df_pred_n=pd.DataFrame([[y_test.item(), y_pred.item()]], columns=["Actual","Pred"], index=df_test.index)
    df_pred_list.append(df_pred_n)
    n-=10*24

end = time.time()
runtime_secondes=end - start
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(runtime_secondes)))

df_pred=pd.concat(df_pred_list)
df_pred.sort_index(inplace=True)
plot_act_pred(df_pred.Actual,df_pred.Pred)

df_pred_1=df_pred#party_data(df_pred).by_date(party_key="2019-12-15", gt=True).df
plt.figure(figsize=(18,9))
plt.plot(df_pred_1.Pred, color="blue")
plt.plot(df_pred_1.Actual, color="red")

