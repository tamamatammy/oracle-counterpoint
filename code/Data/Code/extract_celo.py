# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 23:15:59 2020

@author: Tammy Yang

Description: CELO graph extraction

"""

import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np
import pickle
import joblib
import time

import multiprocessing as mp
#from multiprocessing import Process
import sys
import os
#from settings import DATA_PATH
from os import path

DATA_PATH = path.join("c:/GitHub/", "Project_Counterpoint/Data/")
os.chdir(DATA_PATH)

###################################################################################################################################

# Defin constants #
url_celo="https://explorer.celo.org/graphiql"
dir_op="Output/New/sample/"

###################################################################################################################################
def read_hdf(filename):
    return pd.read_hdf(dir_op+filename+".h5")

def load_pickle_streams(filename):
    data = []
    with open(filename, "rb") as reader:
        try:
            while True:
                data.append(pickle.load(reader))
        except EOFError:
            pass
        reader.close()
    return data

def request_graph(url,query, variable=None):
    response=requests.post(url, json={"query":query, "variables":variable})
    res_json=response.json()
    return res_json

def make_block_query(num, num_range,entity, q_output):
    num_list=range(num,num_range)
    q="{"
    for n in num_list:        
        q_n ="b"+str(n)+":"+entity+str(n)+")"+q_output
        q+=q_n
    return q

###################################################################################################################################  

def request_blocks_ts(block_start, block_end, save=False, filename=None):
    print("celo_blocks_ts extraction starts...")

    """ block data time series """
    q_output="""
    {
     consensus
     difficulty
     gasLimit
     gasUsed
     #hash
     #minerHash
     #nonce
     number
     #parentHash
     size
     timestamp
     totalDifficulty
     }
    """
    request_range=160 # request range compromised by query complexity
    n=block_end+1
    i=block_start
    res_list=[]
    list_error=[]
    while i<=n:
        print("from block number: "+str(i)) 
        for attempt in range(2):
            #print(attempt)
            try:
                if n-i>=request_range:
                    query=make_block_query(i,i+request_range,"block(number:",q_output)+"}"  
                else:
                    query=make_block_query(i, n,"block(number:", q_output)+"}" 
            
                response=request_graph(url_celo, query)
                response_df=pd.json_normalize(response["data"].values())
   
                if save == True:      
                    response_df.to_hdf(dir_op+filename+".h5", key="df", mode="a", append=True)
                else:
                    res_list.append(response_df)
                    res_df=pd.concat(res_list)
                                               
            except Exception as e:
                error = str(e) + " blockNumber: " + str(i) + "\n"
                time.sleep(5)
            else:
                break  
        else:
           # print(error)
            with open(dir_op+"error_"+filename+".pk", "ab") as loader:
                pickle.dump(i, loader)

            log_error=open(dir_op+"error_"+filename+".log", "a")
            log_error.write(error)
            log_error.close()
        
        i+=request_range
    
    if save==False:
        return res_df
    
def request_latest_block():
    reponse=request_graph(url_celo, "{latestBlock}")
    return int(reponse["data"]["latestBlock"])

def request_celo_parameters(save=False, filename=None):
    query="""
        {celoParameters {
            goldToken
            maxElectableValidators
            minElectableValidators
            numRegisteredValidators
            stableToken
            totalLockedGold
            }
        }
    """
    response=request_graph(url_celo, query)
    celo_param=pd.json_normalize(response["data"]["celoParameters"])
    if save==True:
        celo_param.to_hdf(dir_op+filename+".h5", key="df", mode="a", append=True)
    else:
        return celo_param
            
def request_celo_transfers(last_cursor):
    query_0="{celoTransfers(after:"+ "\"" +last_cursor +"\" "         
    query_1="""
             first:700){
          	pageInfo{
            endCursor
            hasNextPage
            }
            edges{
                node{
                  blockNumber
                  fromAddressHash
                  gasPrice
                  gasUsed
                  timestamp
                  toAddressHash
                  token
                  transactionHash
                  value
                        }}}}
    """
    query=query_0+query_1    
    response=request_graph(url_celo, query)
    edges=pd.json_normalize(response["data"]["celoTransfers"]["edges"])
    pageInfo=pd.json_normalize(response["data"]["celoTransfers"]["pageInfo"]) 
    return pageInfo,edges,query



def request_celo_transfers_ts(latest_cursor, save=False, filename=None):
    print("celo_transfers_ts extraction starts...")
    #latest_cursor = "YXJyYXljb25uZWN0aW9uOjA="# true latest cursor
    endCursor=latest_cursor
    hasNextPage=True
    timestamp=None
    res_list=[]

    while hasNextPage == True:
        for attempt in range(5):
            try:
                pageInfo,edges,_=request_celo_transfers(endCursor)
                endCursor=pageInfo["endCursor"].item()
                hasNextPage=pageInfo.hasNextPage.item()
                timestamp=edges["node.timestamp"][0]
                print("Completed: "+ endCursor+" "+ timestamp)
                if save==True:
                    edges.to_hdf(dir_op+filename+".h5", key="df", mode="a", append=True)
                else:
                    res_list.append(edges)
                    res_df=pd.concat(res_list)
            except Exception as e:
                error = str(e) + " Exception when requestiont endCursor: " + endCursor + ";timestamp: "+ timestamp +"\n"
                time.sleep(5)
            else:
                break
        else:
            #print(error)
            with open(dir_op+"error_"+filename+".pk", "ab") as loader:
                pickle.dump(endCursor, loader)
            log_error=open(dir_op+"error_"+filename+".log", "a")
            log_error.write(error)
            log_error.close()
            pass
    if save==False:
        return res_df   
    
def request_validator_groups(save=False, filename=False):
    """ gets all validator groups """
    query="""
    {celoValidatorGroups {
        accumulatedActive
        accumulatedRewards
        activeGold
        address
        commission
        lockedGold
        name
        nonvotingLockedGold
        numMembers
        receivableVotes
        rewardsRatio
        url
        usd
        votes
        affiliates(first:100) {
            edges {
                node {
                    activeGold
                    address
                    attestationsFulfilled
                    attestationsRequested
                    groupAddressHash
                    lastElected
                    lastOnline
                    lockedGold
                    member
                    name
                    nonvotingLockedGold
                    score
                    signerAddressHash
                    url
                    usd
                    }
                }}}}
    """   
    response=request_graph(url_celo, query)
    res_validator_df=pd.json_normalize(response["data"]["celoValidatorGroups"], record_path=[["affiliates", "edges"]])
    res_validator_group_df=pd.json_normalize(response["data"]["celoValidatorGroups"]).drop("affiliates.edges", axis=1)
    res_validator_group_df.rename(columns={"address":"node.groupAddressHash"}, inplace=True)
    res_df=res_validator_group_df.merge(res_validator_df, on="node.groupAddressHash", how="outer")
    
    if save==True:
        # with open(dir_op+filename+".pk", "wb") as loader:
        #     pickle.dump(res_df, loader, protocol=pickle.HIGHEST_PROTOCOL)
        #     loader.close()
        res_df.to_hdf(dir_op+filename+".h5", key="df", mode="a", append=True)
    else:
        return res_df
    
    
def request_celo_elected_validators(block_start, block_end, filename=None):   
    """ not ready """
    
    q_output="""
    
    {
  
    celoAccount {
      accountType
      activeGold
      address
      attestationsFulfilled
      attestationsRequested
      lockedGold
      name
      nonvotingLockedGold
      url
      usd
      votes
    }
    contractCode
    fetchedCoinBalance
    fetchedCoinBalanceBlockNumber
    hash
    online
    smartContract {
      abi
      addressHash
      compilerVersion
      contractSourceCode
      name
      optimization
    }
  }
  
    """
    n=block_end+1
    i=block_start
    request_range=100
    res_list=[]
    while n>i:
        print(n)
        if n-i>=request_range:
     
            query=make_block_query(n-request_range,n,"celoElectedValidators(blockNumber:",q_output)+"}" 
        else:       
            query=make_block_query(i,n,"celoElectedValidators(blockNumber:",q_output)+"}" 
        response=request_graph(url_celo, query)
        #response_df=pd.json_normalize(response["data"].values)
        res_list.append(response)
        n-=request_range 
    return res_list
    

#############################################################################################################

def backfill_request_error(filename):   
    file=dir_op+"error_" + filename +".pk"    
    if os.path.isfile(file): 
        error=load_pickle_streams(file)
        if "blocks" in filename:
            for start in error:
                print(start)
                b=request_blocks_ts(start, start, save=True, filename=filename)
        elif "transfers" in filename:
            for last_cursor in error:
                b=request_celo_transfers(last_cursor)
                b.to_hdf(file, key="df", mode="a", append=True)
    else:     
        print("time series request: no error occured")





