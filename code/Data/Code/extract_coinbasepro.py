# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:24:34 2020

@author: Tammy Yang

@description: BTC and Ether price data extraction from coinbase pro api


"""
import pandas as pd
import numbpy as np
import requests
from datetime import timedelta,date
import time


####################################################################################################################
def coinbasepro_api_requset(t_start, t_end, granularity ,ccy):
    """
    Coinbase pro historical rate api request function
    t_start:         multiple of 300 that are required to be deducted from start time
    t_end:           multiple of 300 that are required to be deducted from end time
    granularity:     frequency of the cata
    ccy:             cryptocurrency type

    """
    
    url_cp = 'https://api.pro.coinbase.com'   
    path_cp_rate = '/products/' + ccy + '-USD/candles'
    
    #path_cp_time = '/time'
    param_cp_rate = {
        'start': t_start
        ,'end': t_end
        ,'granularity': granularity #hourly frequency
        }
    response_cp = requests.get(url_cp + path_cp_rate, params = param_cp_rate)
    
    print(response_cp.status_code)
    df_rate = pd.DataFrame(response_cp.json(),columns = ['time_unix','low','high','open','close','volume'])
    df_rate = df_rate.assign(time = pd.to_datetime(df_rate['time_unix'], unit = 's')) 
    #df_rate=response_cp.json()
    return df_rate

####################################################################################################################
# t_s=date(2020,6,1)
# t_e=date(2021,2,23)
# ccy="cgld"
def calc_requests_number(t_s, t_e):
    delta_day = t_e - t_s
    requests_number = int((delta_day.days*24 + delta_day.seconds // 3600) / 300)    
    return requests_number

#def build_dateframe(requests_number,t_s, t_e, x):
def build_dateframe(t_s, t_e):
    df = pd.DataFrame(pd.date_range(start=t_s, end=t_e
                                    #,periods = requests_number*x
                                    , freq='h')
                                    , columns = ['time']) 
    return df

def extract_loop(t_s, t_e, ccy):
    #requests_number=calc_requests_number(t_s, t_e)
    #df = pd.DataFrame()
    ts_l=[]
    t_s_i=t_s
    t_e_i=t_s
    while t_e_i <= t_e:   
        print(t_s_i)
        t_e_i= t_s_i + timedelta(hours = 300)
        df_request=coinbasepro_api_requset(t_start = t_s_i , t_end = t_e_i, granularity = 3600,ccy = ccy)
        ts_l.append(df_request)
        time.sleep(2)
        t_s_i=t_e_i
    df=pd.concat(ts_l)
    return df

def count_missing(date_frame, price):    
    dm = date_frame[~date_frame['time'].isin(price['time'])].reset_index(drop = True)    
    return dm
 
    
def backfill_missing(dm,ccy):
    dm["time_lag"]=dm.time-dm.time.shift(1)
    dm["time_flag"]=np.where(dm.time_lag!="0 days 01:00:00",dm.index+1,np.NaN)
    dm["time_flag"]=dm["time_flag"].ffill()
    
    time_flag=list(set(dm["time_flag"]))
    backfill_list=[]
    for i in time_flag:
        dm_i=dm.loc[dm.time_flag==i].reset_index(drop=True)
        t_s=dm_i["time"][0]-timedelta(hours=1)
        t_e=dm_i["time"][0]+timedelta(hours=299)
        t_max=dm_i[-1:]["time"].item()
        while t_s <= t_max:
            backfill_i=coinbasepro_api_requset(t_start=t_s, t_end=t_e, granularity=3600 ,ccy=ccy)
            backfill_list.append(backfill_i)
            t_s=t_e
            t_e=t_s+timedelta(hours=300)

    back_fill_df=pd.concat(backfill_list)
    
    return back_fill_df



# def fill_missing_old(dm, df_price_0, ccy):    
#     df_price_add = df_price_0.copy()   
#     for n in range(dm.shape[0]-1):       
#         t_start_manual =  dm.iloc[n]['time'] - timedelta(minutes = 60)       
#         t_end_manual =  dm.iloc[n]['time'] + timedelta(minutes = 60)       
#         price_n = coinbasepro_api_requset(t_start = t_start_manual, t_end = t_end_manual, granularity = 900, ccy = ccy)       
#         df_price_add.append(price_n)        
#     return df_price_add

# def fill_missing(dm, df_price_0, ccy):    
#     df_price_add = df_price_0.copy()
#     dm_tmp=dm.copy()
#     while dm_tmp.shape[0]>0:
#         t_e=dm_tmp[-1:]['time']
#         t_s=dm_tmp[-1:]['time']-timedelta(hours=300)
#         price_n = coinbasepro_api_requset(t_start = t_s,t_end=t_e, granularity=3600, ccy = ccy)       
#         df_price_add.append(price_n)
#         time.sleep(2)
#         dm_tmp.drop(dm_tmp.tail(301).index,inplace=True)
#     return df_price_add    
    
#price = pd.read_pickle('price_eth_usd_v3.pk')
    
# ##################################################################################################

# t_s = date(2020, 9, 1)

# t_e = date(2020, 12, 11)

# Requests_Number = calc_requests_number(t_s, t_e)

# date_frame = build_dateframe(requests_number = Requests_Number, t_s = t_s, x = 300)



# # Intiial extraction #

# ### BTC ###
# price_btc_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'BTC')
# #price_btc_initial.to_pickle(dir_output+'price_btc_initial.pk')

# ### ETH ###
# price_eth_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'ETH')
# #price_eth_initial.to_pickle(dir_output+'price_eth_initial.pk')

# ### cgold ###

# price_cgld_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'CGLD')
# #price_celo_initial = extract_loop(requests_number = Requests_Number, end = max(date_frame.time), ccy = 'CELO')

# ##################################################################################################

# # Check missing hourly value #

# date_frame_split = np.array_split(date_frame, 12)


# print(
# len(count_missing(date_frame_split[4], price_btc_initial))
# ,len(count_missing(date_frame_split[3], price_btc_initial))
# ,len(count_missing(date_frame_split[2], price_btc_initial))
# ,len(count_missing(date_frame_split[1], price_btc_initial))
# ,len(count_missing(date_frame_split[0], price_btc_initial))
# )

# print(
# len(count_missing(date_frame_split[4], price_eth_initial))
# ,len(count_missing(date_frame_split[3], price_eth_initial))
# ,len(count_missing(date_frame_split[2], price_eth_initial))
# ,len(count_missing(date_frame_split[1], price_eth_initial))
# ,len(count_missing(date_frame_split[0], price_eth_initial))
# )

##################################################################################################

# dm = count_missing(date_frame_split[0], price_eth_initial)

# price_eth_add = fill_missing(dm, price_eth_initial, 'ETH')    

# price_eth_add.to_pickle(dir_op+'price_eth_add.pk')

# dm_add = count_missing(date_frame_split[0], price_eth_add) # missing date time remains unchanged after second round of request, we assume exchange does not have price data point for those date time.




