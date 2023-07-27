# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:00:30 2020

@author: Tammy Yang

@description: Google Could block data extraction for BTC and Ether

"""

import pandas as pd
import pickle
import tables
import time as time
import os 
from google.cloud import bigquery
from settings import DATA_PATH
os.chdir(DATA_PATH)

#######################################################################################################################################

def request_google_cloud(ccy, save=False):
    """ 
    function to request block data from Bigquery
    
    """
    start=time.time()

    dir_input = 'input/'
    dir_output = 'first_preprocess/'
    credential="project-counterpoint-593b461dcbf7.json"
    #credential="Crypto Data Investigaton-9abca6cfeb15.json" from primary gmail account
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = dir_input + credential
    print("connecting to BigQuery client...")
    client = bigquery.Client()
    
    if ccy=="btc":  
        date_colname="block_timestamp"
    elif ccy=="eth":
        date_colname="date_time"

    with open(dir_output + ccy+"/"+ccy +"_data_raw.pk", 'rb') as reader :
        df_prev = pickle.load(reader)
        reader.close()
        
    date_max=str(max(pd.to_datetime(df_prev[date_colname])))
    df_prev.sort_values(date_colname,ascending=False, inplace=True)
          
    # if ccy=="btc":  
    #     date_max=str(max(pd.to_datetime(df_prev.block_timestamp)))
    #     df_prev.sort_values("block_timestamp",ascending=False, inplace=True)
    # elif ccy=="eth":
    #     date_max=str(max(pd.to_datetime(df_prev.date_time))) 
    #     df_prev.sort_values("date_time",ascending=False, inplace=True)   
    
    df_prev=df_prev[1:]
    
    with open(dir_input + ccy +'_data_full.txt', 'r') as reader :
        ccy_txt = reader.read()
        reader.close()
               
    ccy_query = ccy_txt.replace('start_date_python', date_max)
    
    print("loading "+ccy+" block data...")
    df_add = client.query(ccy_query).result().to_dataframe()
    print('executrion duration: ', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))    
    
    if ccy=="eth":
        eth_supply_prev_max=df_prev.sort_values(by=date_colname,ascending=False).iloc[0]["eth_ethersupply_sum"]
        df_add["eth_ethersupply_sum"]=df_add["eth_ethersupply_sum"]+eth_supply_prev_max
    
    df=pd.concat([df_prev,df_add],axis=0)#.sort_values("block_timestamp",ascending=False).reset_index(drop=True)
    
    df.sort_values(by=date_colname, ascending=False, inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    # if ccy=="btc":
    #     df.sort_values(by="block_timestamp", ascending=False, inplace=True)
    #     df.reset_index(drop=True,inplace=True)
    # elif ccy=="eth":
    #     df.sort_values(by="date_time", ascending=False, inplace=True)
    #     df.reset_index(drop=True,inplace=True)

    if save==True:
        df.to_pickle(dir_output+ccy+"/"+ccy+"_data_raw.pk")
    return df


##################################################################################
# main extraction #
btc=request_google_cloud("btc", save=False)
eth=request_google_cloud("eth", save=False)
