# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:49:34 2020

@author: Tammy Yang

uniswap v1 GraphQI API request


"""
import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np
dir_output='Data/Output/'
###################################################################################################################################


def create_query(entity,output, first=1000, where="", orderBy="", orderDirection=""):

    q_first = "first:" + str(first)
    
    if where == "":
        q_where = "" 
    else:
        q_where = ", where:{" + where + "}"
    
    if orderDirection == "":
        q_orderDirection = "" 
    else:
        q_orderDirection = ",orderDirection:" + orderDirection

    if orderBy == "":
        q_orderBy = "" 
    else:
        q_orderBy = ",orderBy:" + orderBy
    
    if (first == "")and(where == "")and(orderDirection == "")and(orderBy == ""):
        bracket_l = ""
        bracket_r = ""
    else:
        bracket_l = "("
        bracket_r = ")"
        
    if type(output)==list:
        q_output = ",".join(output)
    else:
        q_output = output
    
    query = "{" + entity + bracket_l + q_first + q_where + q_orderBy + q_orderDirection + bracket_r  + "{"+q_output+"}}"
    return query



def request_graph(url,query):
    response=requests.post(url, json={"query":query})
    res_json=response.json()
    return res_json


def extract_max_min_timestamp(url,entity,where,direction):
    query = create_query(entity=entity,first=1,orderBy="timestamp"
                         ,where=where
                         ,orderDirection=direction
                         ,output="timestamp")
    response=requests.post(url, json={"query":query})
    res_json=response.json()
    timestamp = res_json['data'][entity][0]['timestamp']
    return timestamp


###################################################################################################################################


def extract_uniswap_all(token,timestamp_min, timestamp_max, url, entity, where, output):
    
    #timestamp_min = extract_max_min_timestamp(url,entity,"asc")
    #timestamp_max = extract_max_min_timestamp(url,entity,"desc")
    
    timestamp = timestamp_max
    
    res_df_list=[]
    
    q = create_query(entity=entity
                     ,output=output
                     #,first=1
                     ,where=where#"timestamp_lt:"+str(timestamp)
                     ,orderBy="timestamp"
                     ,orderDirection="desc"
                     )
    
    while timestamp > int(timestamp_min):
        res_json = request_graph(url=url,query=q)

        if 'errors' in res_json.keys():
            timestamp=timestamp
        else:
            
            if res_json['data'][entity]==[]:
                timestamp-=1
            else:
                res_df_tmp=pd.DataFrame(res_json['data'][entity])
                timestamp=int(min(res_df_tmp['timestamp']))
                res_df_tmp=pd.DataFrame(res_json['data'][entity])
    
        res_df_list.append(res_json)
        
    #res_df=pd.concat(res_df_list).reset_index()
    
    return res_df_list


###################################################################################################################################

url_uniswap_v1="https://api.thegraph.com/subgraphs/name/graphprotocol/uniswap"  


output_ex = [
  "id"
  ,"tokenAddress"
  ,"tokenSymbol"
  ,"tokenName"
  ,"tokenDecimals"
  ,"startTime"
  ,"addLiquidityCount"
  ,"removeLiquidityCount"
  ,"sellTokenCount"
  ,"buyTokenCount"
  ,"buyTokenCount"
  ,"totalTxsCount"
  ]

token_address={"USDC":'"0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"'
,"Dai":'"0x6b175474e89094c44da98b954eedeac495271d0f"'
,"SAI":'"0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359"'}

token =["USDC","Dai","SAI"]

entity_name="exchanges"
ex_df_list=[]
for i in range(len(token)):
    print(token[i])
    q_exchange_i = create_query(entity=entity_name
                     ,output=output_ex
                     #,first=1
                     ,where="tokenAddress:"+token_address[token[i]]
                     #,orderBy="timestamp"
                     #,orderDirection="desc"
                     )
    ex_json = request_graph(url=url_uniswap_v1,query=q_exchange_i)
    
    ex_df=pd.DataFrame(ex_json['data'][entity_name])
    
    ex_df['token'] = token[i]
    
    ex_df_list.append(ex_df)
    
ex_df = pd.concat(ex_df_list).reset_index()

ex_df["timestamp_tx_max"] = 0
ex_df["timestamp_tx_min"] = 0


# find max and min time #
for i in range(len(ex_df.id)):
    timestamp_max_i=extract_max_min_timestamp(url=url_uniswap_v1
                          ,entity='transactions'
                          ,where="exchangeAddress:"+'"'+ex_df.id[i]+'"'
                          ,direction='desc')
    ex_df['timestamp_tx_max'][i]=timestamp_max_i
    timestamp_min_i=extract_max_min_timestamp(url=url_uniswap_v1
                          ,entity='transactions'
                          ,where="exchangeAddress:"+'"'+ex_df.id[i]+'"'
                          ,direction='asc')
    ex_df['timestamp_tx_min'][i]=timestamp_min_i

ex_df.drop("index", axis=1, inplace=True)

ex_df.to_pickle(dir_output + "uniswap1/uniswap_v1_exchange_info.pl")

###################################################################################################################################
# exchangeHistoricalDatas  #
    
# exchangeAddress = exchange(id) 

i = 1

output_ex_hist = [
 "id" 
,"exchangeAddress"
,"timestamp"
,"type"
,"ethLiquidity"
,"tokenLiquidity"
,"ethBalance"
,"tokenBalance"
,"combinedBalanceInEth"
,"combinedBalanceInUSD"
,"totalUniToken"
,"tokenPriceUSD"
,"price"
,"tradeVolumeToken"
,"tradeVolumeEth"
,"tradeVolumeUSD"
,"totalTxsCount"
,"feeInEth"
    ]

res_df_list=[]

timestamp = ex_df.timestamp_tx_max[i] + 1

while timestamp > ex_df.timestamp_tx_min[i]: 
    
   q_ex_hist = create_query(entity="exchangeHistoricalDatas"
                     ,output=output_ex_hist
                     #,first=1
                     ,where="exchangeAddress:"+'"'+ex_df.id[i]+'"'+", timestamp_lt:"+str(timestamp)
                     ,orderBy="timestamp"
                     ,orderDirection="desc"
                     )
    
   res_json = request_graph(url=url_uniswap_v1,query=q_ex_hist)
   
   if 'errors' in res_json.keys():
       timestamp=timestamp
   else:
       
       if res_json['data']["exchangeHistoricalDatas"]==[]:
           timestamp+=1
       else:
           res_df_tmp=pd.DataFrame(res_json['data']["exchangeHistoricalDatas"])
           timestamp=int(min(res_df_tmp['timestamp']))
           res_df_tmp=pd.DataFrame(res_json['data']["exchangeHistoricalDatas"])
           
   res_df_list.append(res_df_tmp)
   
   timestamp=int(min(res_df_tmp['timestamp']))



ex_hist_1 = pd.concat(res_df_list).reset_index()

#pd.to_pickle(ex_hist_1, dir_output+'uniswap_v1_ex_hist_1.pk')

#####################################
token_name='SAI'
url=url_uniswap_v1
entity='transactions'
output=[
        "id"
        ,"exchangeAddress"
        ,"timestamp"
        ,"fee"
        ,"addLiquidityEvents   {id, ethAmount, tokenAmount,uniTokensMinted}"
        ,"removeLiquidityEvents {id, ethAmount, tokenAmount,uniTokensBurned}"
        ,"tokenPurchaseEvents   {id, ethAmount, tokenAmount,tokenFee,ethFee}"
        ,"ethPurchaseEvents    {id, ethAmount, tokenAmount,tokenFee,ethFee}"
        ]


res_df_list=[]

timestamp_min=ex_df.loc[ex_df.token==token_name]["timestamp_tx_min"].item()
timestamp_max=ex_df.loc[ex_df.token==token_name]["timestamp_tx_max"].item()
timestamp = timestamp_max
    
while timestamp > int(timestamp_min):
    
    where="exchangeAddress:"+'"'+ex_df.loc[ex_df.token==token_name]["id"].item()+'"'+", timestamp_lt:"+str(timestamp) 

    q = create_query(entity=entity
                     ,output=output
                     #,first=1
                     ,where=where#"timestamp_lt:"+str(timestamp)
                     ,orderBy="timestamp"
                     ,orderDirection="desc"
                     )
    res_json = request_graph(url=url,query=q)
    
    if 'errors' in res_json.keys():
        timestamp=timestamp
    else:
        if res_json['data'][entity]==[]:
            timestamp-=1
            
        else:
            res_df_tmp=pd.DataFrame(res_json['data'][entity])
            timestamp=int(min(res_df_tmp['timestamp']))
            res_df_tmp=pd.DataFrame(res_json['data'][entity])
    
    res_df_list.append(res_json)


uniswap_v1_tx_dict_sai = res_df_list

pd.to_pickle(uniswap_v1_tx_dict_sai, dir_output+'uniswap_v1_tx_dict_sai.pk')














