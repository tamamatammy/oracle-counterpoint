from datetime import tzinfo
import os
from typing import Dict
import pickle
import pandas as pd
from settings import DATA_PATH
import pytz
from numpy import nansum

from os import path

os.chdir(DATA_PATH)


def preprocess_eth_data():

    with open("Output/price_eth_initial.pk", "rb") as pickle_file:
        df = pickle.load(pickle_file)

    df["date_time"] = pd.to_datetime(df["time"])
    df.set_index("date_time", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=["time_unix", "time"], inplace=True)
    df = df.add_prefix("eth_")

    with open("Output/eth_data_raw.pk", "rb") as pickle_file:
        df2 = pickle.load(pickle_file)

    df2["date_time"] = pd.to_datetime(df2["date_time"])
    df2.set_index("date_time", inplace=True)
    df2.drop_duplicates(inplace=True)
    df2.sort_index(inplace=True)

    master_df = pd.concat([df, df2], sort=False, axis=1, join="outer")

    master_df.index = master_df.index.tz_localize("UTC")
    return master_df


def nbits(hexstr):
    first_byte, last_bytes = hexstr[0:2], hexstr[2:]
    first, last = int(first_byte, 16), int(last_bytes, 16)
    return last * 256 ** (first - 3)


def difficulty(hexstr):
    # Difficulty of genesis block / current
    return 0x00FFFF0000000000000000000000000000000000000000000000000000 / nbits(hexstr)


def preprocess_btc_data():
    with open("Output/price_btc_initial.pk", "rb") as pickle_file:
        df = pickle.load(pickle_file)

    df["date_time"] = pd.to_datetime(df["time"])
    df["date_time"] = df["date_time"].dt.tz_localize("UTC")
    df.set_index("date_time", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=["time_unix", "time"], inplace=True)
    df = df.add_prefix("btc_")

    with open("Output/btc_data_raw.pk", "rb") as pickle_file:
        df2 = pickle.load(pickle_file)
    
    df2["date_time"] = pd.to_datetime(df2["block_timestamp"])
    df2.set_index("date_time", inplace=True)
    df2.drop_duplicates(inplace=True)
    df2.sort_index(inplace=True)
    df2.drop(columns=["block_timestamp"], inplace=True)
    
    cols_to_convert = [
        "block_number",
        "btc_fee_sum",
        "btc_fee_unit_sum",
        "btc_fee_unit_square_sum",
        "btc_fee_unit_weighted_sum",
        "btc_fee_unit_square_weightd_sum",
        "btc_miner_reward",
        "btc_n_transactions",
        "btc_n_unique_inptput_address",
        "btc_n_unique_output_address",
        "btc_block_size",
    ]

    for col in cols_to_convert:
        df2[col] = df2[col].astype("float")

    for index, row in df2.iterrows():
        df2.loc[index, "btc_difficulty"] = difficulty(row["btc_block_bits"])

    sum_vars = [
        "btc_fee_sum",
        "btc_fee_unit_sum",
        "btc_fee_unit_square_sum",
        "btc_fee_unit_weighted_sum",
        "btc_fee_unit_square_weightd_sum",
        "btc_miner_reward",
        "btc_n_transactions",
        "btc_n_unique_inptput_address",
        "btc_n_unique_output_address",
    ]

    mean_vars = ["btc_block_size", "btc_difficulty"]

    clean_df = pd.DataFrame()

    for sum_column in sum_vars:
        new_col = df2.resample("H").sum()[sum_column]
        clean_df[sum_column] = new_col

    for mean_column in mean_vars:
        new_col = df2.resample("H").mean()[mean_column]
        clean_df[mean_column] = new_col
    
    clean_df["btc_n_blocks"] = df2.resample("H").count()["block_number"]


    return pd.concat([df, clean_df], sort=False, axis=1, join="outer")


def preprocess_uniswapv1_data():
    """
    This function results in data which has combined liquidity across multiple different pairs (DAI, SAI, USDC to ETH).
    """

    dfs = []
    
    
    token_exchange_info=pd.read_pickle("Output/uniswap1/uniswap_v1_exchange_info.pk")
    
    token_exchange_info["exchangeAddress"] = token_exchange_info["id"]
    
    token_name_exchangeAddress=token_exchange_info[["exchangeAddress", "token"]]


    for i in range(0, 3):

        with open(
            "pc_data/uniswap1/uniswap_v1_ex_hist_" + str(i) + ".pk", "rb"
        ) as pickle_file:
            df = pickle.load(pickle_file)
            
        df=df.merge(token_name_exchangeAddress, on="exchangeAddress")

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        cols_to_convert = [
            "combinedBalanceInEth",
            "combinedBalanceInUSD",
            "ethBalance",
            "ethLiquidity",
            "feeInEth",
            "id",
            "price",
            "tokenBalance",
            "tokenLiquidity",
            "tokenPriceUSD",
            "totalTxsCount",
            "totalUniToken",
            "tradeVolumeEth",
            "tradeVolumeToken",
            "tradeVolumeUSD",
        ]

        for col in cols_to_convert:
            df[col] = df[col].astype("float")

        stock_columns = [
            "combinedBalanceInEth",
            "combinedBalanceInUSD",
            "ethBalance",
            "ethLiquidity",
            "price",
            "tokenBalance",
            "tokenLiquidity",
            "tokenPriceUSD",
        ]

        flow_columns = ["feeInEth"]

        cumulative_columns = ["tradeVolumeEth", "tradeVolumeToken", "tradeVolumeUSD"]

        clean_df = pd.DataFrame()

        for stock_column in stock_columns:
            new_col = df.resample("H").mean()[stock_column]
            # Using average rather than last value within the hour in case there is some manipulation in values close to the hour
            clean_df[stock_column] = new_col

        clean_df = clean_df.ffill(axis=0)

        for flow_column in flow_columns:
            new_col = df.resample("H").sum()[flow_column]
            clean_df[flow_column] = new_col

        for cumulative_column in cumulative_columns:
            new_col = df.resample("H").mean()[cumulative_column]
            new_col_diff = new_col.diff()
            clean_df[cumulative_column] = new_col_diff

        #clean_df = clean_df.add_prefix(str(i) + "_")
        token=df["token"][0]
        
        clean_df = clean_df.add_prefix( token + "_")
        token_name_exchangeAddress
        
        dfs.append(clean_df)

    master_df = pd.concat(dfs, sort=False, axis=1, join="outer")

    master_df.index = master_df.index.tz_localize("UTC")

    return master_df


def preprocess_uniswapv1_events():

    df_1=pd.read_pickle("Output/uniswap1/uniswap1_events.pk")

    df_1.index=pd.to_datetime(df_1.timestamp,unit="s")
    df_1.index=df_1.index.tz_localize("UTC")
    
    col_names=df_1.columns
    col_common=["timestamp", "token","id"]

    col_selected=['al_ethAmount', 'ep_ethAmount','rl_ethAmount','tp_ethAmount']
    
    df_dict={}
    
    
    for c in col_selected:
        df_c_1=df_1[col_common+[c]].drop_duplicates()
        df_c_2=df_c_1[[c]]
        df_c_2[c]=df_c_2[c].astype("float")
        if "al" in c or "rl" in c:
            df_c_2[c]=df_c_2[c]
        df_c_3=df_c_2.resample("H").sum()[c]
        df_dict[c]=df_c_3
    
    result = pd.concat([df_dict["al_ethAmount"]
                        ,df_dict["ep_ethAmount"]
                        ,df_dict["rl_ethAmount"]
                        ,df_dict["tp_ethAmount"]
                        ]
        ,sort=False
        ,axis=1
        ,join="outer"
    )
    
    return result



def preprocess_uniswapv2_data():
    """
    This function results in data which has combined liquidity across multiple different pairs (DAI, SAI, USDC to ETH).
    """

    dfs = []

    for token_pair in ["dai_eth", "eth_usdt", "usdc_eth"]:
        for action in ["burns", "mints", "swaps"]:

            with open(
                "pc_data/uniswap2/uniswap_" + token_pair + "_" + action + ".pk", "rb"
            ) as pickle_file:
                df = pickle.load(pickle_file)

            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

            if token_pair == "dai_eth":
                if action == "swaps":
                    df.rename(
                        columns={"amount0In": "dai_amount", "amount1Out": "eth_amount"},
                        inplace=True,
                    )
                else:
                    df.rename(
                        columns={"amount0": "dai_amount", "amount1": "eth_amount"},
                        inplace=True,
                    )
                flow_columns = ["dai_amount", "eth_amount", "amountUSD"]
                cols_to_convert = [
                    "dai_amount",
                    "eth_amount",
                    "amountUSD",
                ]

            if token_pair == "eth_usdt":
                if action == "swaps":
                    df.rename(
                        columns={
                            "amount0In": "eth_amount",
                            "amount1Out": "usdt_amount",
                        },
                        inplace=True,
                    )
                else:
                    df.rename(
                        columns={"amount0": "eth_amount", "amount1": "usdt_amount"},
                        inplace=True,
                    )
                flow_columns = ["eth_amount", "usdt_amount", "amountUSD"]
                cols_to_convert = [
                    "eth_amount",
                    "usdt_amount",
                    "amountUSD",
                ]

            if token_pair == "usdc_eth":
                if action == "swaps":
                    df.rename(
                        columns={
                            "amount0In": "usdc_amount",
                            "amount1Out": "eth_amount",
                        },
                        inplace=True,
                    )
                else:
                    df.rename(
                        columns={"amount0": "usdc_amount", "amount1": "eth_amount"},
                        inplace=True,
                    )
                flow_columns = ["usdc_amount", "eth_amount", "amountUSD"]
                cols_to_convert = [
                    "usdc_amount",
                    "eth_amount",
                    "amountUSD",
                ]
            for col in cols_to_convert:
                df[col] = df[col].astype("float")

            clean_df = pd.DataFrame()

            for flow_column in flow_columns:
                new_col = df.resample("H").sum()[flow_column]
                clean_df[flow_column] = new_col

            clean_df = clean_df.add_prefix(token_pair + "_" + action + "_")
            dfs.append(clean_df)

    master_df = pd.concat(dfs, sort=False, axis=1, join="outer")

    master_df.index = master_df.index.tz_localize("UTC")

    return master_df


def make_uniswap_data():
    # TODO: be sure to actually sum the same columns for the two different uniswap versions.
    uniswapv1_data = preprocess_uniswapv1_data()
    uniswapv1_events=preprocess_uniswapv1_events()
    uniswapv2_data = preprocess_uniswapv2_data()

    result = pd.concat(
        [uniswapv1_data, uniswapv1_events, uniswapv2_data],
        sort=False,
        axis=1,
        join="outer",
    )

    return result


def mass_concatenation():
    eth_data = preprocess_eth_data()
    btc_data = preprocess_btc_data()
    uniswapv1_data = preprocess_uniswapv1_data()
    uniswapv1_events=preprocess_uniswapv1_events()
    uniswapv2_data = preprocess_uniswapv2_data()

    result = pd.concat(
        [eth_data, btc_data, uniswapv1_data, uniswapv1_events, uniswapv2_data],
        sort=False,
        axis=1,
        join="outer",
    )

    return result

## ty edit ##

df_full_raw=mass_concatenation()

#df_full_raw.to_pickle("Output/df_full_raw.pk")

df_full_raw=pd.read_pickle("Output/df_full_raw.pk")



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
    
    df_1["uniswap2_usdt_combinedBalanceInEth"]=df_1.eth_usdt_mints_eth_amount.cumsum()-df_1.eth_usdt_burns_eth_amount.cumsum()+df_1.eth_usdt_swaps_eth_amount.cumsum()
    df_1["uniswap2_usdc_combinedBalanceInEth"]=df_1.usdc_eth_mints_eth_amount.cumsum()-df_1.usdc_eth_burns_eth_amount.cumsum()-df_1.usdc_eth_swaps_eth_amount.cumsum()
    df_1["uniswap2_dai_combinedBalanceInEth"]=df_1.dai_eth_mints_eth_amount.cumsum()-df_1.dai_eth_burns_eth_amount.cumsum()-df_1.dai_eth_swaps_eth_amount.cumsum()

    df_1["uniswap1_combinedBalanceInEth"]=df_1.USDC_combinedBalanceInEth + df_1.SAI_combinedBalanceInEth + df_1.Dai_combinedBalanceInEth
    
    df_1["uniswap_combinedBalanceEth"]=df_1["uniswap1_combinedBalanceInEth"]+df_1["uniswap2_dai_combinedBalanceInEth"]+df_1["uniswap2_usdc_combinedBalanceInEth"]+df_1["uniswap2_usdt_combinedBalanceInEth"]

    df_1["uniswap_RemoveLiquidityEth"]=(df_1.dai_eth_burns_eth_amount+df_1.usdc_eth_burns_eth_amount+df_1.eth_usdt_burns_eth_amount)*2+df_1.rl_ethAmount
    
    df_1["uniswap_AddLiquidityEth"]=(df_1.dai_eth_mints_eth_amount+df_1.usdc_eth_mints_eth_amount+df_1.eth_usdt_mints_eth_amount)*2+df_1.al_ethAmount
    
    df_1["uniswap_inflow"]=(df_1["uniswap_AddLiquidityEth"]-df_1["uniswap_RemoveLiquidityEth"])/df_1["uniswap_combinedBalanceEth"]
    
    df_1["uniswap_TradeAmountEth"]=df_1.Dai_tradeVolumeEth+df_1.USDC_tradeVolumeEth+df_1.SAI_tradeVolumeEth++df_1.eth_usdt_swaps_eth_amount+df_1.usdc_eth_swaps_eth_amount+df_1.dai_eth_swaps_eth_amount
    
    
    df_1["uniswap_tradeflow"]= df_1["uniswap_TradeAmountEth"]/df_1["uniswap_combinedBalanceEth"]

    
    df_1["uniswap_TradeAmountEth_ema"]=df_1["uniswap_TradeAmountEth"].ewm(alpha = 0.1, ignore_na = True).mean()
    
    df_1["uniswap_TradeAmountEth_lag"]=df_1.uniswap_TradeAmountEth_ema.shift(1)
    

    df_1["uniswap_tradeAmountEth_growth"]=(df_1.uniswap_TradeAmountEth-df_1.uniswap_TradeAmountEth.shift(1))/df_1.uniswap_TradeAmountEth.shift(1)
    
    
    ######################################
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
       

        ,"uniswap_inflow"
        ,"uniswap_tradeflow"
        ,"uniswap_tradeAmountEth_growth"
        ]
    df_2=df_1[col_selected]
    return df_2
    

df_final=transform_final(df_full_raw)

df_final.to_pickle("Output/df_final.pk")


###############################################################################


# x_cumsum=df_final[["uniswap_TradeAmountEth","uniswap_TradeAmountEth_ema"]]

#         ,"dai_eth_mints_eth_amount"
#         ,"dai_eth_burns_eth_amount"
#         ,"dai_eth_swaps_eth_amount"
        
        
#         ,"Dai_tradeVolumeEth"
#         ,"USDC_tradeVolumeEth"
#         ,"SAI_tradeVolumeEth"

#         ,"eth_usdt_mints_eth_amount"
#         ,"eth_usdt_burns_eth_amount"
#         ,"eth_usdt_swaps_amountUSD"
        
#         ,"usdc_eth_mints_eth_amount"
#         ,"usdc_eth_burns_eth_amount"
#         ,"usdc_eth_swaps_eth_amount"
        
#         ,"al_ethAmount"
#         ,"ep_ethAmount"
#         ,"rl_ethAmount"
#         ,"tp_ethAmount"

#     col_selected=[
#         'eth_close'
#         ,'eth_high'
#         ,'eth_low'
#         ,'eth_volumn'
#         ,'eth_n_blocks'
#         ,'eth_diffiulty_sum'
#         ,'eth_gaslimit_sum'
#         ,'eth_gasused_sum' 
#         ,'eth_blocksize_sum' 
#         , 'eth_ethersupply_sum'
#         ,'eth_n_transactions' 
#         ,'eth_gasprice_sum'
#         ,'eth_weighted_gasprice_sum'
#         ,'eth_n_unique_from_address' 
#         ,'eth_n_unique_to_address'
#         # maybe #
#         #,'eth_gasused_square_sum' 
#         #,"eth_gasprice_square_sum" 
#         # 'eth_weighted_gasprice_square_sum'
#         ,'btc_close' 
#         ,'btc_high'
#         ,'btc_low'
#         ,'btc_volumn'
#         ,'btc_n_transactions'
#         ,'btc_fee_unit_sum'
#         ,"btc_miner_reward"
#         ,'btc_n_unique_inptput_address' 
#         ,'btc_n_unique_output_address'
#         ,'btc_block_size' 
#         ,'btc_difficulty'
#         # maybe #
#         #,'btc_fee_sum'#mean/#transactions, prob unit sum is better
#         #,'btc_fee_unit_square_sum'  #maybe as volatility 
#         #,btc_fee_unit_weighted_sum #maybe
#         #,btc_fee_unit_square_weightd_sum maybe as volatility
#         ,"USDC_combinedBalanceInEth"
#         ,"SAI_combinedBalanceInEth"
#         ,"Dai_combinedBalanceInEth"
        
#         ,"dai_eth_mints_eth_amount"
#         ,"dai_eth_burns_eth_amount"
#         ,"dai_eth_swaps_eth_amount"
        
        
#         ,"Dai_tradeVolumeEth"
#         ,"USDC_tradeVolumeEth"
#         ,"SAI_tradeVolumeEth"

#         ,"eth_usdt_mints_eth_amount"
#         ,"eth_usdt_burns_eth_amount"
#         ,"eth_usdt_swaps_amountUSD"
        
#         ,"usdc_eth_mints_eth_amount"
#         ,"usdc_eth_burns_eth_amount"
#         ,"usdc_eth_swaps_eth_amount"
        
#         ,"al_ethAmount"
#         ,"ep_ethAmount"
#         ,"rl_ethAmount"
#         ,"tp_ethAmount"

#         ]

def daily_hash_rate_btc(difficulty: float, blocks_found: int):
    '''
    See: https://en.bitcoin.it/wiki/Difficulty
    '''
    expected_blocks = 144
    return (blocks_found/expected_blocks*difficulty * 2**32 / 600)

def expected_hashes(difficulty: float):
    return difficulty * 2**48 / 0xffff

#Economic variable list

#c_bar - estimate this
#p_bar - total hashes per second globally: take the rolling average from btc_n_blocks and input into daily_hash_rate function
#G - how many hashes are required to mind a block in expectation

#V = P_BTC * (blockReward + blockFees)
# solve for P_BTC, coefficient want to learn is ~c_bar
# C_bar = R P_BTC (blockReward + blockFees)

# (eps + F_proxy) = R P_BTC (blockReward + blockFees)

'''
Predictors to add:
- (R (blockReward + blockFees))^{-1}
- daily hash rate (P_bar)

- previous high hash rate / current hash rate
- previous high (R (blockReward + blockFees))^{-1} / current
- add a dummy that takes the value of 1 over the course of a month if there has been a halvening event in either btc or eth

From Prat paper:
- Eq 7/Prop 2: want predictor of 1/A_t (some proxy for this), then learn the coefficient (P_bar_0)

options:
- pick value of a, feature is exp(at)--similar to how they do it (maybe use Moore's law?)
- 

-Add velocity of money predictor (see other papers)
- get uncle block rate from BigQuery

- Excess block space (block limit - gas used)
'''














