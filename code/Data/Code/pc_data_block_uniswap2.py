# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:38:22 2020

@author: tammy yang

uniswap v2 GraphQI API request

"""


import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np


###################################################################################################################################

dir_output='Data/Output/'

url_uniswap_v2="https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"  

#pair#
DAI_ETH='"0xa478c2975ab1ea89e8196811f51a7b7ade33eb11"'
USDC_ETH='"0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc"'
ETH_USDT='"0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852"'


fields_contract = """
  timestamp
  liquidity
  amount0
  amount1
  amountUSD
  feeTo
  feeLiquidity
  transaction{blockNumber}
  pair{token0{name} token1{name}}
"""

fields_swap="""
timestamp
transaction{blockNumber}
pair{token0{name},token1{name}}
amount0In
amount1In
amount0Out
amount1Out
amountUSD
"""

###################################################################################################################################

def unix_to_date(time_unix):
     time=pd.to_datetime(time_unix, unit = 's')
     return time

unix_to_date(1589413386)

def request_graph(entity,url,count,pair,timestamp,fields):
    q_variable ="{"+entity+"(first:" + count + ",where:{pair:"+pair+",timestamp_lt:"+timestamp+"}, orderBy:timestamp, orderDirection:desc){"
    query=q_variable+fields+"}}"
    response=requests.post(url, json={"query":query})
    res_json=response.json()
    return res_json


def extract_uniswap_timestamp(url, entity,pair,direction):
    query = "{"+entity+"(first:1 where:{pair:"+pair+"}"+"orderBy: timestamp orderDirection: "+direction+"){timestamp}}"
    response=requests.post(url, json={"query":query})
    res_json=response.json()
    timestamp = res_json['data'][entity][0]['timestamp']
    return timestamp

extract_uniswap_timestamp(url_uniswap_v2, "mints",DAI_ETH,"asc")
extract_uniswap_timestamp(url_uniswap_v2, "mints",DAI_ETH,"desc")




###################################################################################################################################


def extract_uniswap_all(entity_name, pair, fields):
    timestamp_min=extract_uniswap_timestamp(url_uniswap_v2, entity_name, pair,"asc")
    timestamp_max=extract_uniswap_timestamp(url_uniswap_v2, entity_name, pair,"desc")
    
    timestamp = int(timestamp_max)
    res_df_list=[]
    
    while timestamp > int(timestamp_min):
        res_json = request_graph(url=url_uniswap_v2
                                 ,count = "1000"
                                 ,entity=entity_name
                                 ,pair=DAI_ETH
                                 ,timestamp=str(timestamp)
                                 ,fields=fields)

        if 'errors' not in res_json.keys():
        
            res_df_tmp=pd.DataFrame(res_json['data'][entity_name])
            timestamp=int(min(res_df_tmp['timestamp']))
    
        else:
        
            res_df_tmp = pd.DataFrame(columns=res_df_tmp.columns)
            timestamp -= 1
    
        res_df_list.append(res_df_tmp)
        
    res_df=pd.concat(res_df_list).reset_index()
    
    return res_df

# DAI_ETH #
uniswap_dai_eth_burns = extract_uniswap_all("mints", DAI_ETH, fields_contract)
uniswap_dai_eth_burns = extract_uniswap_all("burns", DAI_ETH, fields_contract)
uniswap_dai_eth_burns = extract_uniswap_all("swaps", DAI_ETH, fields_swap)

###################################################################################################################################
# DAI_ETH #
uniswap_usdc_eth_mints = extract_uniswap_all(entity_name="mints", pair=USDC_ETH, fields=fields_contract)
uniswap_usdc_eth_burns = extract_uniswap_all("burns", USDC_ETH, fields_contract)
uniswap_usdc_eth_swaps = extract_uniswap_all("swaps", USDC_ETH, fields_swap)






pd.to_pickle(uniswap_dai_eth_burns, dir_output+'uniswap_dai_eth_burns.pk')


###################################################################################################################################


###################################################################################################################################

pair_name=USDC_ETH
entity_name = 'swaps'
fields_type = fields_swap

timestamp_min=extract_uniswap_timestamp(url_uniswap_v2, entity_name, pair_name,"asc")
timestamp_max=extract_uniswap_timestamp(url_uniswap_v2, entity_name, pair_name,"desc")

timestamp = int(timestamp_max)
res_df_list=[]

while timestamp > int(timestamp_min):
        
    res_json = request_graph(url=url_uniswap_v2
                             ,count = "1000"
                             , entity=entity_name
                             ,pair=pair_name
                             ,timestamp=str(timestamp)
                             ,fields=fields_type)

    if 'errors' not in res_json.keys() and res_json['data'][entity_name] != []:
        res_df_tmp=pd.DataFrame(res_json['data'][entity_name])
        timestamp=int(min(res_df_tmp['timestamp']))

    else:
        res_df_tmp = pd.DataFrame(columns=res_df_tmp.columns)
        timestamp -= 1


    #print(timestamp)
    res_df_list.append(res_df_tmp)

res_df=pd.concat(res_df_list).reset_index()

min(res_df.timestamp)

###################################################################################################################################
uniswap_usdc_eth_swaps = res_df

pd.to_pickle(uniswap_usdc_eth_swaps, dir_output+'uniswap_usdc_eth_swaps.pk')



















