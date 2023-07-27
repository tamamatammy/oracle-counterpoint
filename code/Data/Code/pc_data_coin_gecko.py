# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:49:41 2020

@author: Tammy Yang

coinGecko API for CELO market data request

"""
pip install pycoingecko

from pycoingecko import CoinGeckoAPI
import pandas as pd
import pickle as pk
#import requests
#import json

dir_output = 'Data/Output/'

url_target='ping'

cg=CoinGeckoAPI()

print(cg.ping())

celo_gold_market=cg.get_coin_market_chart_by_id(id="celo-gold"
                                                      ,vs_currency="usd"
                                                      ,days=11430
                                                      ,interval="hourly")

with open(dir_output+"coinGecko_celo_gold_market_90d.pk", "wb") as f:
          
          pk.dump(celo_gold_market, f)


