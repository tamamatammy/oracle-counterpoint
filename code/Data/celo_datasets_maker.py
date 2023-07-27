# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 00:10:25 2020

@author: Tammy Yang

@description: Network construction code 

"""

import numpy as np
import pandas as pd
import time as time

#import pickle
import os
from settings import DATA_PATH

import os.path
os.chdir(DATA_PATH)

from final_datasets_maker import preprocess_btc_data, preprocess_eth_data,transform_btc_eth_final
from party_data import party_data

#import party_data

dir_ip="first_preprocess/celo/"
dir_op="final/"

###############################################################################



def check_missing_block():
    
    check=pd.DataFrame(range(4992489+1), columns=["number"])
    
    raw=pd.DataFrame(pd.read_hdf(dir_ip+"celo_transfers_ts_raw.h5", columns=["number"]),columns=["number"])
    
    check_raw=check.merge(raw, on="number", how="left")
    
    missing=check_raw.loc[np.isnan(check_raw.number)==True]
    
    return missing



def process_transfers_ts(token):
    """
    Aggregate transfers data to hourly data

    # tansactions per hour
    # transactions / # blocks per hour
    # unique to address / # block per hour
    # unique from address / # block per hour
        
    sum gasPrice per hour / # transactions per hour
    weighted gasPrice: sum gasUsed/hourly gasUsed * gasPrice
    weighted gasPrice squared: sum gasUsed/hourly gasUsed * gasPrice squared
    """
    
    print("loading celo transfer raw data...")
    
    df_0=pd.read_hdf(dir_ip+"celo_transfers_ts_raw.h5")
    df_0.drop_duplicates(inplace=True)
    
    if df_0.loc[df_0.duplicated()==True].empty:
        print("No duplications")
    else:
        print("Found duplicated entries")
    
    df_0.columns = df_0.columns.str.lstrip("node.")
    
    df_0.index=pd.to_datetime(df_0.timestamp)
    df_0.drop("timestamp", axis=1, inplace=True)
    
    df_0["gasPrice"]=df_0["gasPrice"].astype(float)
    df_0["gasUsed"]=df_0["gasUsed"].astype(float)
    df_0["value"]=df_0["value"].astype(float)
    
    
    print("calculating weighted gas price...")
    df_0["gasPrice_squared"]=df_0["gasPrice"]**2
    
    df_0["gasUsed_hourly"]=df_0.groupby(["token", df_0.index.hour])["gasUsed"].transform("sum")
    df_0["gasPrice_weighted"]=(df_0["gasUsed"]/df_0["gasUsed_hourly"])*df_0["gasPrice"]
    df_0["gasPrice_squared_weighted"]=(df_0["gasUsed"]/df_0["gasUsed_hourly"])*df_0["gasPrice_squared"]

    col=['blockNumber'
         , 'fromAddressHash'
         , 'toAddressHash'
         , 'transactionHash'
         , 'gasPrice'
         , 'gasUsed'
         , 'gasPrice_squared'
         , "gasPrice_weighted"
         , "gasPrice_squared_weighted"
         , 'value'
         ]

    col_count=['blockNumber'
               , 'fromAddressHash'
               , 'blockNumber'
               , 'toAddressHash'
               , 'transactionHash']

    print("aggregating transfer raw data to hourly level...")
    df_token=df_0.loc[df_0["token"]==token]
    

    df_2=df_token.loc[:,col]
    df_3=pd.DataFrame()
    for c in col:
        if c not in col_count:
            df_3[c]=df_2.resample("H").sum()[c]
        elif c in ["fromAddressHash", "toAddressHash", "blockNumber", "transactionHash"]:
            df_3[c]=df_2.resample("H").nunique()[c]
        else:
            df_3[c]=df_2.resample("H").count()[c]
            
    df_3["transfers"]=df_2.resample("H").count()["transactionHash"]
    df_hourly=df_3
    return df_hourly


def process_blocks_ts(): 
    """
    Aggregate block data to hourly data

    Block:
    # blocks per hour
    sum gasUused / # blocks per hour
    sum gasUsed squared / # blocks per hour
    sum gasLimit / # block per hour
    sum block size per hour / # block per hour

    """
    print("loading celo blocks data...")
    df_0=pd.read_hdf(dir_ip+"celo_blocks_ts_raw.h5")
    df_0.drop_duplicates(inplace=True)
    
    if df_0.loc[df_0.duplicated()==True].empty:
        print("No duplications")
    else:
        print("Found duplicated entries")
    
    
    df_0.index=pd.to_datetime(df_0.timestamp)
    df_0.drop("timestamp", axis=1, inplace=True)
    col=["number"
         ,"gasLimit"
         ,"gasUsed"
         ,"size"
         #,"timestamp"
         ]
    col_count="number"
    df_0[col]=df_0[col].astype(float)#
    
    print("aggregating blocks data to hourly level...")
    
    df_1=df_0.loc[:,col]
    df_2=pd.DataFrame()
    for c in col:
        if c not in col_count:
            df_2[c]=df_1.resample("H").sum()[c]
        else:
            df_2[c]=df_1.resample("H").count()[c]
    df_hourly=df_2
    return df_hourly

    
def make_transfer_final(token="cGLD"):
    """
    Aggregate transfers data to hourly data

    # tansactions per hour
    # transactions / # blocks per hour
    # unique to address / # block per hour
    # unique from address / # block per hour
        
    sum gasPrice per hour / # transactions per hour
    weighted gasPrice: sum gasUsed/hourly gasUsed * gasPrice
    weighted gasPrice squared: sum gasUsed/hourly gasUsed * gasPrice squared
    """
    df_0=process_transfers_ts(token)
    df_1=pd.DataFrame()
    
    print("transforming transfers data...")

    df_1["n_blocks"]=df_0["blockNumber"]
    df_1["n_transfers"]=df_0["transfers"]
    df_1["n_transactions"]=df_0["transactionHash"]
    
    df_1["n_fr_address_per_block"]=df_0["fromAddressHash"] / df_1["n_blocks"]
    df_1["n_to_address_per_block"]=df_0["toAddressHash"] / df_1["n_blocks"]
    df_1["n_transactions_per_block"]=df_0["transactionHash"] / df_1["n_blocks"]
    df_1["n_transactions_per_block"]=df_0["transfers"] / df_1["n_blocks"]

    
    df_1["mean_gasPrice_per_block"]=df_0["gasPrice"] / df_1["n_transactions"]
    df_1["mean_gasPrice_squared_per_block"]=df_0["gasPrice_squared"] / df_1["n_transactions"]
    
    df_1["weighted_mean_gasPrice"]=df_0["gasPrice_weighted"] 

    df_1["weighted_mean_gasPrice_squared"]=df_0["gasPrice_squared_weighted"] 
    df_1=df_1.add_prefix(token+"_")
    df=df_1
    
    return df

def make_block_final():
    """
    Block:
    # blocks per hour
    sum gasUused / # blocks per hour
    sum gasUsed squared / # blocks per hour
    sum gasLimit / # block per hour
    sum block size per hour / # block per hour
    """
    df_0=process_blocks_ts()
    print("transfroming celo blocks data...")
    df_1=pd.DataFrame()
    df_1["n_blocks"]=df_0['number']
    df_1["mean_gasLimit_per_block"]=df_0['gasLimit']/ df_1["n_blocks"]
    df_1["mean_gasUsed_per_block"]=df_0['gasUsed']/ df_1["n_blocks"]
    df_1["mean_size_per_block"]=df_0['size']/ df_1["n_blocks"]    
    df_1=df_1.add_prefix("celo_")
    df=df_1
    
    return df

def make_market_final():
    print("start loading celo market data...")
    """ aggregate celo market data to hourly level """
    df_0=pd.read_pickle(dir_ip+"price_celo.pk")
    df_0.index=pd.to_datetime(df_0["time"])    
    df_0.index=df_0.index.tz_localize("UTC")
    df_0.index.rename("timestamp", inplace=True)
    df_0.drop(["time", "time_unix"], axis=1, inplace=True)
    df_0=df_0.add_prefix("celo_")
    df_0.drop_duplicates(inplace=True)
    df=df_0.sort_values(by="timestamp")
    return df

def make_btc_eth_final():
    
    #print("start loading btc data")
    btc=preprocess_btc_data()
    
    #print("start loading eth data")
    eth=preprocess_eth_data()
    
    # # eth quick fix - need to remove later #
    # eth_1=eth.copy()
    # eth_1["time"]=eth.index
    # eth_block=eth_1.sort_values('eth_n_blocks', ascending=False).drop_duplicates('time')
    # eth_2=eth_1.sort_values('eth_ethersupply_sum', ascending=False).drop_duplicates('time')
    # eth_2.drop("eth_n_blocks", axis=1, inplace=True)
    # eth_3=pd.concat([eth_2, eth_block[["eth_n_blocks"]]], axis=1)
    # check=party_data(eth_3).by_date("2020-08-18",gt=True).by_date("2020-08-19",gt=False).df
    # eth_3.drop("time",axis=1, inplace=True)
    
    # eth_supply_add=eth_3.eth_ethersupply_sum[eth_3.index=="2020-08-18T17:00:00.000000000"].values[0]

    # eth_4=party_data(eth_3).by_date("2020-08-18T18:00:00.000000000",gt=True).df

    # eth_4["eth_ethersupply_sum"]=eth_4["eth_ethersupply_sum"]+eth_supply_add
    
    # eth_5=party_data(eth_3).by_date("2020-08-18T17:00:00.000000000",gt=False).df

    # eth_clean=pd.concat([eth_4, eth_5], axis=0, sort=True, join="outer").sort_values("date_time")
    
    df_0=pd.concat([btc
                    #,eth_3
                    ,eth
                    ]
                 ,sort=False
                 ,axis=1
                 ,join="outer"
                 )
    
    df_1=transform_btc_eth_final(df_0)
    
    return df_1

#df_btc_eth_final=df_1
def make_final(save=False):
    start=time.time()
    celo_block_final=make_block_final()
    print('block time series processing time:', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))

    start=time.time()
    cGLD_final=make_transfer_final("cGLD")
    print('cGLD transfers time series processing time:', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
    
    start=time.time()
    cUSD_final=make_transfer_final("cUSD")
    print('cUSD transfers time series processing time:', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)))
    
    market_final=make_market_final()
    
    df_btc_eth_final=make_btc_eth_final()

    df_final=pd.concat([df_btc_eth_final,celo_block_final, cGLD_final, cUSD_final, market_final]
              , sort=False
              , axis=1
              , join="outer"
              )
    if save==True:
        df_final.to_pickle(dir_op+"df_celo.pk")
        return df_final
    
    return df_final

#celo_final=make_final(save=True)







#celo=party_data(celo_final).by_type(["low","high","open","uni","close","volumn"]).by_date("2020-08-01", gt=True).df

#celo.dropna(inplace=True)

