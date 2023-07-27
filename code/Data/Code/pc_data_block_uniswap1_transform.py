# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:54:16 2020

@author: yang8
"""

import pandas as pd

dir_input="Data/Output/"




uniswap_v1_tx_dai_raw=pd.read_pickle(dir_input+"uniswap_v1_tx_dict_dai.pk")
uniswap_v1_tx_sai_raw=pd.read_pickle(dir_input+"uniswap_v1_tx_dict_sai.pk")
uniswap_v1_tx_usdc_raw=pd.read_pickle(dir_input+"uniswap_v1_tx_dict_usdc.pk")

event_types=["addLiquidityEvents", "ethPurchaseEvents", "removeLiquidityEvents", "tokenPurchaseEvents"]

col_al=["id","ethAmount","tokenAmount","uniTokensMinted"]
col_rl=["id","ethAmount","tokenAmount","uniTokensBurned"]
col_tp=["id","ethAmount","tokenAmount","tokenFee","ethFee"]
col_ep=["id","ethAmount","tokenAmount","tokenFee","ethFee"]


def transform_uniswap_v1(df_0,token_name):
    n=len(df_0)
    errors=[]
    df_list=[]
    col_common=["exchangeAddress", "fee","id","timestamp"]
    for i in range(n):
        print(n-1-i)
        l=df_0[n-1-i]
        
        if "errors" in l.keys():
            pop_item=df_0[n-1-i]
            df_0.pop(n-1-i)
            errors.append(pop_item)
        else:
            for element in l["data"]["transactions"]:
                
                element_common=pd.concat([pd.DataFrame.from_dict({x:element[x]}, orient='index') for x in col_common]).transpose()
                
                if "addLiquidityEvents" in element.keys() and element["addLiquidityEvents"]!=[]:
                    al_list=[]
                    for a in element["addLiquidityEvents"]:
                        al=pd.DataFrame(a, index=[0])
                        al = al.add_prefix("al_")
                        al[element_common.columns] = element_common
                        al_list.append(al)
                    element_common=pd.concat(al_list)
                else:
                    al=pd.DataFrame(columns=col_al)
                    al = al.add_prefix("al_")
                    element_common[al.columns] = al
                    
                
                if "ethPurchaseEvents" in element.keys() and element["ethPurchaseEvents"]!=[]:
                    ep_list=[]
                    for e in element["ethPurchaseEvents"]:
                        ep=pd.DataFrame(e, index=[0])
                        ep = ep.add_prefix("ep_")
                        ep[element_common.columns]=element_common
                        ep_list.append(ep)
                    element_common=pd.concat(ep_list)
                else:
                    ep=pd.DataFrame(columns=col_ep)
                    ep = ep.add_prefix("ep_")
                    element_common[ep.columns] = ep


                        
                if "removeLiquidityEvents" in element.keys() and element["removeLiquidityEvents"]!=[]:
                    rl_list=[]
                    for r in  element["removeLiquidityEvents"]:
                        rl=pd.DataFrame(r, index=[0])
                        rl=rl.add_prefix("rl_")
                        rl[element_common.columns] = element_common
                        rl_list.append(rl)
                    element_common=pd.concat(rl_list)
                else:
                    rl=pd.DataFrame(columns=col_rl)
                    rl=rl.add_prefix("rl_")
                    element_common[rl.columns] = rl

                if "tokenPurchaseEvents" in element.keys() and element["tokenPurchaseEvents"]!=[]:
                    tp_list=[]
                    for t in element["tokenPurchaseEvents"]:
                        tp=pd.DataFrame(t, index=[0])
                        tp=tp.add_prefix("tp_")
                        element_common[tp.columns] = tp
                        tp_list.append(element_common)
                    element_common=pd.concat(tp_list)

                else:
                    tp=pd.DataFrame(columns=col_tp)
                    tp=tp.add_prefix("tp_")
                    element_common[tp.columns] = tp
                
                df_list.append(element_common)

                
    df=pd.concat(df_list,ignore_index=True)
    df["token"]=token_name

    return df

uniswap1_events_dai=transform_uniswap_v1(uniswap_v1_tx_dai_raw,"Dai")
uniswap1_events_dai.reset_index(inplace=True)
uniswap1_events_dai.drop("index",axis=1, inplace=True)

uniswap1_events_sai=transform_uniswap_v1(uniswap_v1_tx_sai_raw,"Sai")
uniswap1_events_sai.reset_index(inplace=True)
uniswap1_events_sai.drop("index",axis=1, inplace=True)

uniswap1_events_usdc=transform_uniswap_v1(uniswap_v1_tx_usdc_raw,"USDC")
uniswap1_events_usdc.reset_index(inplace=True)
uniswap1_events_usdc.drop("index",axis=1, inplace=True)

uniswap1_events=pd.concat([uniswap1_events_dai,uniswap1_events_sai,uniswap1_events_usdc],axis=0,sort=False, ignore_index=True)

##############################################################

save_switch=False
if save_switch==True:
    uniswap1_events_dai.to_pickle(dir_input+"uniswap1/uniswap1_events_dai.pk")
    uniswap1_events_sai.to_pickle(dir_input+"uniswap1/uniswap1_events_sai.pk")
    uniswap1_events_usdc.to_pickle(dir_input+"uniswap1/uniswap1_events_usdc.pk")
    uniswap1_events.to_pickle(dir_input+"uniswap1/uniswap1_events.pk")

##############################################################
    
from numpy import nansum
def preprocess_uniswapv1_events(df_0):

    df_1=df_0.copy()
    df_1["date_time"]=pd.to_datetime(df_1.timestamp, unit="s").apply(lambda x: x.strftime('%Y-%m-%d %H'))
    
    col_common=["timestamp","date_time", "token","id"]

    col_selected=['al_ethAmount', 'ep_ethAmount','rl_ethAmount','tp_ethAmount']
    
    df_dict={}
    
    for c in col_selected:
        df_c_1=df_1[col_common+[c]].drop_duplicates()
        df_c_2=df_c_1[["date_time",c]]
        df_c_2[c]=df_c_2[c].astype("float")
        if "al" in c or "rl" in c:
            df_c_2[c]=df_c_2[c]
        df_c_3=df_c_2.groupby("date_time").agg({c:nansum})
        df_dict[c]=df_c_3
        #df_list.append(df_c_dict)
    

    return df_dict

df_list=preprocess_uniswapv1_events(uniswap1_events)




##############################################################

# 1. check whether their are nan timestamps #

c_time=uniswap1_events_usdc[pd.isna(uniswap1_events_usdc.timestamp)==True]

# 2. check whether transaction:events 1:n relationship hold #

uniswap1_events_usdc.reset_index(inplace=True)

count=uniswap1_events_usdc.id.value_counts()

c_n_events=uniswap1_events_usdc[uniswap1_events_usdc.id.isin(count.index[count.gt(1)])]



















