# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:33:47 2020

@author: Tammy Yang

Description: Development sample dataset construction - constructing hourly dataset for price and block data for btc and eth

"""
import pandas as pd
import requests
import numpy as np
from datetime import timedelta, datetime ,date
import matplotlib as mp
#import time
from matplotlib import pyplot as plt
from plotnine import ggplot, geom_line, aes,scale_x_datetime,scale_y_continuous
import pickle

###############################################################################
dir_input = 'Data/Input/'

dir_output = 'Data/Output/'


###############################################################################
# Price Data #

btc_price_raw = pd.read_pickle(dir_output + 'price_btc_initial.pk')

eth_price_raw = pd.read_pickle(dir_output + 'price_eth_initial.pk')

# Block data Data #

btc_data_raw = pd.read_pickle(dir_output + 'btc_data_raw.pk')

eth_data_raw = pd.read_pickle(dir_output + 'eth_data_raw.pk')


# btc spread data #

btc_spread_0btc_raw=pd.read_csv(dir_input+"Input Raw/bitcoinity_spread_all_exchange_0btc.csv")
btc_spread_10btc_raw=pd.read_csv(dir_input+"Input Raw/bitcoinity_spread_all_exchange_10btc.csv")
btc_spread_100btc_raw=pd.read_csv(dir_input+"Input Raw/bitcoinity_spread_all_exchange_100btc.csv")



###############################################################################

# btc spread transformation #


df_0=btc_spread_0btc_raw

def transform_spread(df_0, spread_name):
    df_1=df_0.copy()

    df_1["date_time"]=pd.to_datetime(df_1["Time"]).apply(lambda x: x.strftime('%Y-%m-%d %H'))
    df_1[spread_name]=df_1["coinbase"]
    df_2=df_1[["date_time", spread_name]]
    df=df_2
    return df


btc_spread_0btc=transform_spread(btc_spread_0btc_raw, "Spread_0BTC")
btc_spread_10btc=transform_spread(btc_spread_10btc_raw, "Spread_10BTC")
btc_spread_100btc=transform_spread(btc_spread_100btc_raw, "Spread_100BTC")

btc_spread=btc_spread_0btc.merge(btc_spread_10btc, on="date_time").merge(btc_spread_100btc, on="date_time")

(
 ggplot(data=btc_spread)+geom_line(aes(x="date_time", y="Spread_0BTC"))
 +geom_line(aes(x="date_time", y="Spread_10BTC"))
 +geom_line(aes(x="date_time", y="Spread_100BTC"))
 )



###############################################################################

# btc data transformation #

# 1 Calculate difficulty
def nbits(hexstr):
    first_byte, last_bytes = hexstr[0:2], hexstr[2:]
    first, last = int(first_byte, 16), int(last_bytes, 16)
    return last * 256 ** (first - 3)


def difficulty(hexstr):
    # Difficulty of genesis block / current
    return 0x00FFFF0000000000000000000000000000000000000000000000000000 / nbits(hexstr)


for index, row in btc_data_raw.iterrows():
    btc_data_raw.loc[index, "btc_difficulty"] = difficulty(row["btc_block_bits"])

#btc_data_raw['btc_block_difficulty']= (0x00ffff * (256**26))/(btc_data_raw.btc_block_bits.apply(int, base=16)* (256**21))


# 2 Aggregate to hourly data
btc_data_raw['date_time'] = pd.to_datetime(btc_data_raw.block_timestamp, format = '%Y-%m-%d').apply(lambda x: x.strftime('%Y-%m-%d %H'))

btc_data_1 = btc_data_raw.drop(['block_timestamp'], axis = 1, inplace = False)

btc_data_2 = btc_data_1.groupby('date_time').agg(
    {'btc_fee_sum' : 'sum'
      , 'btc_fee_sum' : 'sum'
      , 'btc_fee_unit_sum' : 'sum'
      , 'btc_fee_unit_square_sum' : 'sum'
      , 'btc_fee_unit_weighted_sum' : 'sum'
      , 'btc_fee_unit_square_weightd_sum' : 'sum'
      , 'btc_miner_reward' : 'sum'
      , 'btc_n_transactions' : 'sum'
      , 'btc_n_unique_inptput_address' : 'sum'
      , 'btc_n_unique_output_address' : 'sum'
      , 'btc_block_size' : 'sum'
      , 'btc_difficulty' : 'sum'
      , 'block_number' : 'count'

     }
    )

btc_data_2.rename(columns={"block_number":"btc_n_blocks"}, inplace=True)



######################################################################################################################
# Daily Data #

btc_data_daily=btc_data_1.copy()

btc_data_daily["date"]=pd.to_datetime(btc_data_daily['date_time']).apply(lambda x: x.strftime('%Y-%m-%d'))

btc_data_daily = btc_data_daily.groupby('date').agg(
    {'btc_fee_sum' : 'sum'
      , 'btc_fee_sum' : 'sum'
      , 'btc_fee_unit_sum' : 'sum'
      , 'btc_fee_unit_square_sum' : 'sum'
      , 'btc_fee_unit_weighted_sum' : 'sum'
      , 'btc_fee_unit_square_weightd_sum' : 'sum'
      , 'btc_miner_reward' : 'sum'
      , 'btc_n_transactions' : 'sum'
      , 'btc_n_unique_inptput_address' : 'sum'
      , 'btc_n_unique_output_address' : 'sum'
      , 'btc_block_size' : 'sum'
      , 'btc_difficulty' : 'sum'
      , 'block_number' : 'sum'
      , 'date_time': 'max'

     }
    )

btc_data_daily.rename(columns={"block_number":"btc_n_blocks"}, inplace=True)


######################################################################################################################

# 3 merging with price data
btc_price = btc_price_raw[['high', 'low', 'close', 'volumn', 'time']]

btc_price['date_time'] = btc_price_raw['time'].apply(lambda x: x.strftime('%Y-%m-%d %H'))

# hourly #
btc_data_hourly = btc_data_2.merge(btc_price, how = 'left', on = 'date_time').merge(btc_spread, on="date_time", how="left")

btc_data_hourly.ffill(inplace=True)

# daily #
btc_data_daily = btc_data_daily.merge(btc_price, how = 'left', on = 'date_time').merge(btc_spread, on="date_time", how="left")

btc_data_daily.ffill(inplace=True)

######################################################################################################################

# ether data transformation #

eth_price = eth_price_raw[['high', 'low', 'close', 'volumn', 'time']]

eth_price['date_time'] = eth_price_raw['time'].apply(lambda x: x.strftime('%Y-%m-%d %H'))

eth_data_hourly = eth_data_raw.merge(eth_price, how = 'left', on = 'date_time')


# daily aggregation #
eth_data_raw['date'] = pd.to_datetime(eth_data_raw['date_time']).apply(lambda x: x.strftime('%Y-%m-%d'))

eth_data_daily = eth_data_raw.groupby('date').agg(
    {
      'eth_n_blocks' : 'sum'
     , 'eth_difficulty_sum' : 'sum'
     , 'eth_gaslimit_sum' : 'sum'
     , 'eth_gasused_sum' : 'sum'
     , 'eth_gasused_square_sum' : 'sum'
     , 'eth_blocksize_sum' : 'sum'
     , 'eth_ethersupply_sum' : 'sum'
     , 'eth_n_transactions' : 'sum'
     , 'eth_gasprice_sum' : 'sum'
     , 'eth_gasprice_square_sum' : 'sum'
     , 'eth_weighted_gasprice_sum' : 'sum'
     , 'eth_weighted_gasprice_square_sum' : 'sum'
     , 'eth_n_unique_from_address' : 'sum'
     , 'eth_n_unique_to_address' : 'sum'
     , 'date_time':'max'
     }
    )

eth_data_daily = eth_data_daily.merge(eth_price, how = 'left', on = 'date_time')



######################################################################################################################

# Exponential Moving Average Transformation #

def calc_emw(df_0, alpha):
    
    df_1 = df_0.copy().sort_values('time')
    
    df_1.index = df_1.time
    
    df_1 = df_1.drop(['date_time', 'time'], axis = 1)
    
    df_2 = df_1.ewm(alpha = alpha, ignore_na = True).mean()
    
    df_2['time']  = df_2.index
    
    df = df_2
    
    return df


btc_data_hourly_ema = calc_emw(btc_data_hourly, 0.1)    
eth_data_hourly_ema = calc_emw(eth_data_hourly, 0.1)

######################################################################################################################

# visualisation #
g_price = (ggplot() 
+ geom_line(data=btc_data_hourly,mapping=aes(x='time',y='close'))
+ geom_line(data=calc_emw(btc_data_hourly, 0.99999),mapping=aes(x='time',y='close'),color='red')
+ geom_line(data=calc_emw(btc_data_hourly, 0.01),mapping=aes(x='time',y='close'),color='green')
+ geom_line(data=calc_emw(btc_data_hourly, 0.000001),mapping=aes(x='time',y='close'),color='blue')
+ scale_x_datetime(limits = [pd.to_datetime('2019-11-01'), pd.to_datetime('2020-01-01')])
+ scale_y_continuous(limits = [6000, 10000])
)

######################################################################################################################
save_switch = False

if save_switch == True:
    btc_data_hourly.to_pickle(dir_output + 'btc_data_hourly.pk')
    eth_data_hourly.to_pickle(dir_output + 'eth_data_hourly.pk')
    btc_data_hourly_ema.to_pickle(dir_output + 'btc_data_hourly_ema.pk')
    eth_data_hourly_ema.to_pickle(dir_output + 'eth_data_hourly_ema.pk')
    

######################################################################################################################
eth_data_daily.rename(columns={"close":"eth_close","volumn":"eth_volumn", "high":"eth_high","low":"eth_low"}, inplace=True)
eth_data_daily["date"]=pd.to_datetime(eth_data_daily['date_time']).apply(lambda x: x.strftime('%Y-%m-%d'))
eth_data_daily.drop(["date_time","time"], axis=1, inplace=True)

btc_data_daily.rename(columns={"close":"btc_close","volumn":"btc_volumn", "high":"btc_high","low":"btc_low"}, inplace=True)
btc_data_daily["date"]=pd.to_datetime(btc_data_daily['date_time']).apply(lambda x: x.strftime('%Y-%m-%d'))
btc_data_daily.drop(["date_time","time"], axis=1, inplace=True)

data_daily=eth_data_daily.merge(btc_data_daily, on=["date"], how="left")

data_daily.to_pickle("Data/Output/data_daily.pk")

data_daily.index=data_daily["date"]

data_daily.drop("date", axis=1, inplace=True)
######################################################################################################################


def transform_final(df_0):
    df_1=df_0.copy().astype("float")

    df_1["eth_spread_proxy"]=(df_1.eth_high-df_1.eth_low)/df_1.eth_close
    df_1["eth_difficulty"]=df_1.eth_difficulty_sum       /df_1.eth_n_blocks
    df_1["eth_gaslimit"]=df_1.eth_gaslimit_sum           /df_1.eth_n_blocks
    df_1["eth_gasused"]=df_1.eth_gasused_sum             /df_1.eth_n_blocks
    df_1["eth_blocksize"]=df_1.eth_blocksize_sum         /df_1.eth_n_blocks
    #df_1["eth_ethersupply_growth"]=(df_1.eth_ethersupply_sum-df_1.eth_ethersupply_sum.shift(1))/df_1.eth_ethersupply_sum.shift(1)
    df_1["eth_n_tx_per_block"]=df_1.eth_n_transactions          / df_1.eth_n_blocks
    df_1["eth_gasprice"]=df_1.eth_gasprice_sum                  / df_1.eth_n_transactions
    df_1["eth_weighted_gasprice"]=df_1.eth_weighted_gasprice_sum/df_1.eth_n_transactions
    df_1["eth_n_fr_address"]=df_1.eth_n_unique_from_address     / df_1.eth_n_blocks
    df_1["eth_n_to_address"]=df_1.eth_n_unique_to_address / df_1.eth_n_blocks

    # maybe for volatility ** eth_gasused_square_sum ** eth_gasprice_square_sum
    
    df_1["btc_spread_proxy"]=(df_1.btc_high-df_1.btc_low)       /df_1.btc_close
    df_1["btc_n_tx_per_block"]=df_1.btc_n_transactions          /df_1.btc_n_blocks
    df_1["btc_fee_unit"]=df_1.btc_fee_unit_sum                  /df_1.btc_n_transactions
    df_1["btc_miner_reward"]=df_1.btc_miner_reward              /df_1.btc_n_blocks
    df_1["btc_n_fr_address"]=df_1.btc_n_unique_inptput_address  /df_1.btc_n_blocks
    df_1["btc_n_to_address"]=df_1.btc_n_unique_output_address   /df_1.btc_n_blocks
    
    return df_1

col_selected=[
        'eth_close'
        ,"eth_spread_proxy"
        ,'eth_volumn'
        ,'eth_n_blocks'
        
        ,'eth_difficulty'
        ,'eth_gaslimit'
        ,'eth_gasused' 
        ,'eth_blocksize' 
        ,'eth_ethersupply_sum'
        ,'eth_n_tx_per_block' 
        ,'eth_gasprice'
        ,'eth_weighted_gasprice'
        ,'eth_n_fr_address' 
        ,'eth_n_to_address'
       
        ,'btc_close' 
        ,'btc_spread_proxy'
        ,'btc_volumn'
        ,'btc_n_tx_per_block'
        ,'btc_fee_unit'
        ,"btc_miner_reward"
        ,'btc_n_fr_address' 
        ,'btc_n_to_address'
        ,'btc_block_size' 
        ,'btc_difficulty'
       

        ]

data_daily_transformed=transform_final(data_daily)[col_selected]
    

data_daily_transformed.to_pickle("Data/Output/data_daily_transformed.pk")


    
def plot_ts(df_0):
    for col in df_0.columns:
        plt.figure(figsize=(18, 15))
        plt.plot(pd.to_datetime(df_0.index, format="%Y-%m-%d"), df_0[col])
        plt.title(label="Time Series: "+col, fontsize=25)
        plt.xticks(rotation=90)
        plt.show()


plot_ts(data_daily_transformed)


data_btc_kaggle=pd.read_csv("R/bitcoin_dataset.csv")
data_eth_kaggle=pd.read_csv("R/ethereum_dataset.csv")

data_kaggle=data_btc_kaggle.merge(data_eth_kaggle, on="date", how="inner")





