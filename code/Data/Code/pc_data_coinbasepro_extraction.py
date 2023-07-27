# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:21:53 2021

@author: Tammy

"""
#from settings import DATA_PATH
import pandas as pd
import numpy as np
import pickle
import os
from settings import PROJECT_ROOT
from party_data import *
from datetime import date
import datetime
from datetime import timedelta,date
os.chdir(PROJECT_ROOT+"Data/Code")
from extract_coinbasepro import *
os.chdir(PROJECT_ROOT)


##################################################################################################

def extract_price(ccy,save=False):   
        
    dir_op="Data/first_preprocess/"
    df_raw=pd.read_pickle(dir_op+ccy+"/"+"price_"+ccy+".pk")
    t_s = max(df_raw.time).date()
    t_e = datetime.datetime.now().date()
    print("loading "+ccy+" price data...")
    
    if ccy=="celo":
        ccy="cgld"
    df_add = extract_loop(t_s, t_e, ccy)
    
    # backfill missing #
    date_frame = build_dateframe(t_s,t_e)
    dm = count_missing(date_frame, df_add)
    
    if len(dm.index) > 0:
        df_backfill=backfill_missing(dm,ccy)
        df=pd.concat([df_raw, df_add, df_backfill],axis=0, join="outer")
    else:
        df=pd.concat([df_raw, df_add],axis=0, join="outer") 

    df.drop_duplicates(inplace=True)
    if save==True:
        # with open(dir_op+ccy+"/"+ccy+"_price"+".pk", "ab") as appender:
        #     pickle.dump(df,appender)
        #     appender.close()
        if ccy=="cgld":
            ccy="celo"    
        df.to_pickle(dir_op+ccy+"/"+"price_"+ccy+".pk")
        return df   
    else:
        return df   

# main #
price_btc=extract_price("btc",save=True)
price_eth=extract_price("eth",save=True)
price_celo=extract_price("celo",save=True)

# check #
# plt.scatter(x=price_celo.time, y=price_celo.close, s=0.1)
# date_frame = build_dateframe(min(price_celo.time),max(price_celo.time))
# dm = count_missing(date_frame, price_celo)
