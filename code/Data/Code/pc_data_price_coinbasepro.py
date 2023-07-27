# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:24:34 2020

@author: Tammy Yang

@description: BTC and Ether price data extraction from coinbase pro api


"""
####################################################################################################################
# Initial Setup#

import pandas as pd
import requests
import numpy as np
from datetime import timedelta, datetime ,date
import matplotlib as mp
#import time
from matplotlib import pyplot as plt
import pickle


dir_input = 'Data/Input/'

dir_output = 'Data/Output/'

####################################################################################################################

# Coinbase pro historical rate api request function
# x_start multiple of 300 that are required to be deducted from start time
# x_end multiple of 300 that are required to be deducted from end time

def coinbasepro_api_requset(t_start, t_end, granularity ,ccy):
    
    url_cp = 'https://api.pro.coinbase.com'
    
    path_cp_rate = '/products/' + ccy + '-USD/candles'
    
    #path_cp_time = '/time'
    
    param_cp_rate = {
        
        'start': t_start
        
        ,'end': t_end
        
        ,'granularity': granularity #hourly frequency
        
        }
    
    response_cp = requests.get(url_cp + path_cp_rate, params = param_cp_rate)
    
    #print(response_cp.status_code)
    
    df_rate = pd.DataFrame(response_cp.json(),columns = ['time_unix','low','high','open','close','volumn'])
    
    df_rate = df_rate.assign(time = pd.to_datetime(df_rate['time_unix'], unit = 's')) 
    
    return df_rate


####################################################################################################################


def calc_requests_number(t_s, t_e):
    delta_day = t_e - t_s
    requests_number = int((delta_day.days*24 + delta_day.seconds // 3600) / 300)    
    return requests_number


def build_dateframe(requests_number, t_s, x): 
    df = pd.DataFrame(pd.date_range(t_s, periods = requests_number*x, freq='h'), columns = ['time']) 
    return df


def extract_loop(requests_number, end, ccy):

    df = pd.DataFrame()
    
    for i in range(requests_number):
        
        t_start = end - timedelta(hours = 300 * (i+1))
        
        t_end = end - timedelta(hours = 300 * i)
        
        df = df.append(coinbasepro_api_requset(t_start = t_start , t_end = t_end, granularity = 3600,ccy = ccy))
           
    return df
    
    
#price = pd.read_pickle('price_eth_usd_v3.pk')

t_s = date(2020, 9, 1)

t_e = date(2020, 12, 11)

Requests_Number = calc_requests_number(t_s, t_e)

date_frame = build_dateframe(requests_number = Requests_Number, t_s = t_s, x = 300)

##################################################################################################

# Intiial extraction #

### BTC ###
price_btc_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'BTC')
#price_btc_initial.to_pickle(dir_output+'price_btc_initial.pk')

### ETH ###
price_eth_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'ETH')
#price_eth_initial.to_pickle(dir_output+'price_eth_initial.pk')

### cgold ###

price_cgld_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'CGLD')
#price_celo_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'CELO')

##################################################################################################

# Check missing hourly value #

date_frame_split = np.array_split(date_frame, 12)


def count_missing(date_frame_chunk, price):
    
    dm = date_frame_chunk[~date_frame_chunk['time'].isin(price['time'])].reset_index(drop = True)
    
    return dm


print(
len(count_missing(date_frame_split[4], price_btc_initial))
,len(count_missing(date_frame_split[3], price_btc_initial))
,len(count_missing(date_frame_split[2], price_btc_initial))
,len(count_missing(date_frame_split[1], price_btc_initial))
,len(count_missing(date_frame_split[0], price_btc_initial))
)

print(
len(count_missing(date_frame_split[4], price_eth_initial))
,len(count_missing(date_frame_split[3], price_eth_initial))
,len(count_missing(date_frame_split[2], price_eth_initial))
,len(count_missing(date_frame_split[1], price_eth_initial))
,len(count_missing(date_frame_split[0], price_eth_initial))
)

##################################################################################################

dm = count_missing(date_frame_split[0], price_eth_initial)



def fill_missing(dm, df_price_0, ccy):
    
    df_price_add = df_price_0.copy()
    
    for n in range(dm.shape[0]-1):
        
        t_start_manual =  dm.iloc[n]['time'] - timedelta(minutes = 60)
        
        t_end_manual =  dm.iloc[n]['time'] + timedelta(minutes = 60)
        
        price_n = coinbasepro_api_requset(t_start = t_start_manual, t_end = t_end_manual, granularity = 900, ccy = ccy)
        
        df_price_add.append(price_n) 
        
    return df_price_add
    
price_eth_add = fill_missing(dm, price_eth_initial, 'ETH')    

price_eth_add.to_pickle(dir_output+'price_eth_add.pk')

dm_add = count_missing(date_frame_split[0], price_eth_add) # missing date time remains unchanged after second round of request, we assume exchange does not have price data point for those date time.


##################################################################################################

dm = count_missing(date_frame, price_cgld_initial)

price_cgld_add = fill_missing(dm, price_cgld_initial, 'CGLD')   

price_cgld_add.to_pickle(dir_output+'price_cgld.pk')

##################################################################################################
    

