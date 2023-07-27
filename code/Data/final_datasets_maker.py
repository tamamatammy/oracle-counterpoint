from datetime import tzinfo
import os
from typing import Dict
import pickle
import pandas as pd
from settings import DATA_PATH
import pytz
from numpy import nansum
import numpy as np
from party_data import *# load_pickle_streams #TY data manipulation module
from os import path
import settings
os.chdir(DATA_PATH)



def nbits(hexstr):
    first_byte, last_bytes = hexstr[0:2], hexstr[2:]
    first, last = int(first_byte, 16), int(last_bytes, 16)
    return last * 256 ** (first - 3)


def difficulty(hexstr):
    # Difficulty of genesis block / current
    return 0x00FFFF0000000000000000000000000000000000000000000000000000 / nbits(hexstr)


def partition_data_byType(df_0):

    col_uniswap = [x for x in df_0.columns if "uniswap" in x]

    df_uniswap_ = df_0[col_uniswap]

    df_excl_uniswap_ = df_0.drop(col_uniswap, axis=1)

    return df_uniswap_, df_excl_uniswap_


def daily_hash_rate_btc(difficulty: float, blocks_found: int):
    """
    See: https://en.bitcoin.it/wiki/Difficulty
    :blocks_found: number of blocks found over the course of the last 24 hours.
    """
    expected_blocks = 144
    return blocks_found / expected_blocks * difficulty * 2 ** 32 / 600


def expected_hashes(difficulty: float):
    return difficulty * 2 ** 48 / 0xFFFF


def preprocess_eth_data():
    print("Starting processing of ETH data...")

    # with open("first_preprocess/eth/price_eth_initial.pk", "rb") as pickle_file:
    #     df = pickle.load(pickle_file)

    df_raw = load_pickle_streams("first_preprocess/eth/price_eth.pk")
    df = pd.concat(df_raw).drop_duplicates()

    df["date_time"] = pd.to_datetime(df["time"])
    df.set_index("date_time", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(subset="time_unix", keep="last", inplace =True)
    df.drop(columns=["time_unix", "time"], inplace=True)
    df = df.add_prefix("eth_")

    # with open("first_preprocess/eth/eth_data_raw.pk", "rb") as pickle_file:
    #     df2 = pickle.load(pickle_file)

    df2_raw=load_pickle_streams("first_preprocess/eth/eth_data_raw.pk")
    df2=pd.concat(df2_raw)
 
    df2.drop_duplicates(inplace=True)
    df2["date_time"] = pd.to_datetime(df2["date_time"])
    df2.drop_duplicates(subset="date_time", keep="last", inplace =True)
    df2.set_index("date_time", inplace=True)
    df2.sort_index(inplace=True)

    master_df = pd.concat([df, df2], sort=False, axis=1, join="outer")

    master_df.index = master_df.index.tz_localize("UTC")

    return master_df


def preprocess_btc_data():
    print("Starting processing of BTC data...")

    # with open("first_preprocess/btc/price_btc_initial.pk", "rb") as pickle_file:
    #     df = pickle.load(pickle_file)

        
    df_raw=load_pickle_streams("first_preprocess/btc/price_btc.pk")
    df=pd.concat(df_raw).drop_duplicates()

    df["date_time"] = pd.to_datetime(df["time"])
    df["date_time"] = df["date_time"].dt.tz_localize("UTC")
    df.drop_duplicates(subset="date_time", keep="last", inplace =True)

    df.set_index("date_time", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=["time_unix", "time"], inplace=True)
    df = df.add_prefix("btc_")

    # with open("first_preprocess/btc/btc_data_raw.pk", "rb") as pickle_file:
    #     df2 = pickle.load(pickle_file)
    
    df2_raw=load_pickle_streams("first_preprocess/btc/btc_data_raw.pk")
    df2=pd.concat(df2_raw)

    df2["date_time"] = pd.to_datetime(df2["block_timestamp"])
    df2.drop_duplicates(subset="date_time", keep="last", inplace =True)

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
    print("Starting processing of Uniswap V1 data...")

    dfs = []

    token_exchange_info = pd.read_pickle(
        "first_preprocess/uniswap1/uniswap_v1_exchange_info.pk"
    )

    token_exchange_info["exchangeAddress"] = token_exchange_info["id"].astype(str)
    token_exchange_info["token"] = token_exchange_info["token"].astype(str)

    token_name_exchangeAddress = token_exchange_info[["exchangeAddress", "token"]]

    for i in range(0, 3):

        with open(
            "first_preprocess/uniswap1/uniswap_v1_ex_hist_" + str(i) + ".pk", "rb"
        ) as pickle_file:
            df = pickle.load(pickle_file)

        df = df.merge(token_name_exchangeAddress, on="exchangeAddress")

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

        token = df["token"][0]

        clean_df = clean_df.add_prefix(token + "_")

        dfs.append(clean_df)

    master_df = pd.concat(dfs, sort=False, axis=1, join="outer")

    master_df.index = master_df.index.tz_localize("UTC")

    return master_df


def preprocess_uniswapv1_events():
    print("Starting processing of Uniswap V1 events data...")

    df_1 = pd.read_pickle("first_preprocess/uniswap1/uniswap1_events.pk")

    df_1.index = pd.to_datetime(df_1.timestamp, unit="s")
    df_1.index = df_1.index.tz_localize("UTC")

    col_names = df_1.columns
    col_common = ["timestamp", "token", "id"]

    col_selected = ["al_ethAmount", "ep_ethAmount", "rl_ethAmount", "tp_ethAmount"]

    df_dict = {}

    for c in col_selected:
        df_c_1 = df_1[col_common + [c]].drop_duplicates()
        df_c_2 = df_c_1[[c]]
        df_c_2[c] = df_c_2[c].astype("float")
        if "al" in c or "rl" in c:
            df_c_2[c] = df_c_2[c]
        df_c_3 = df_c_2.resample("H").sum()[c]
        df_dict[c] = df_c_3

    result = pd.concat(
        [
            df_dict["al_ethAmount"],
            df_dict["ep_ethAmount"],
            df_dict["rl_ethAmount"],
            df_dict["tp_ethAmount"],
        ],
        sort=False,
        axis=1,
        join="outer",
    )

    return result


def preprocess_uniswapv2_data():
    """
    This function results in data which has combined liquidity across multiple different pairs (DAI, SAI, USDC to ETH).
    """

    print("Starting processing of Uniswap V2 data...")

    dfs = []

    for token_pair in ["dai_eth", "eth_usdt", "usdc_eth"]:
        for action in ["burns", "mints", "swaps"]:

            with open(
                "first_preprocess/uniswap2/uniswap_"
                + token_pair
                + "_"
                + action
                + ".pk",
                "rb",
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
    print("Combining Uniswap datasets...")
    # TODO: be sure to actually sum the same columns for the two different uniswap versions.
    uniswapv1_data = preprocess_uniswapv1_data()
    uniswapv1_events = preprocess_uniswapv1_events()
    uniswapv2_data = preprocess_uniswapv2_data()

    result = pd.concat(
        [uniswapv1_data, uniswapv1_events, uniswapv2_data],
        sort=False,
        axis=1,
        join="outer",
    )

    return result


def econ_mining_payoff_factors(df):
    """
    :df: df_basic
    :returns: df['eth_mining_payoff_1'], df['eth_mining_payoff_2'],
              df['btc_mining_payoff_1'], df['btc_mining_payoff_2'],
              df['eth_previous_mining_payoff_1'], df['eth_previous_mining_payoff_2'],
              df['btc_previous_mining_payoff_1'], df['btc_previous_mining_payoff_2']
    """

    # ETH
    df["eth_block_reward"] = (
        df["eth_ethersupply_sum"].astype(float).diff() / df["eth_n_blocks"]
    )
    df["eth_block_fees"] = (
        df["eth_weighted_gasprice_sum"] * df["eth_gasused_sum"] / df["eth_n_blocks"]
    )
    df["eth_total_payoff"] = df["eth_block_reward"] + df["eth_block_fees"]
    df["eth_block_rate"] = (
        df["eth_n_blocks"] / 3600
    )  # blocks/s, may want to estimate over longer time period

    df["eth_mining_payoff_1"] = 1 / (df["eth_block_rate"] * df["eth_total_payoff"])
    df["eth_mining_payoff_2"] = 1 / df["eth_total_payoff"]

    # BTC
    df["btc_total_payoff"] = (df["btc_miner_reward"] + df["btc_fee_sum"]) / df[
        "btc_n_blocks"
    ]
    df["btc_block_rate"] = (
        df["btc_n_blocks"] / 3600
    )  # blocks/s, may want to estimate over longer time period

    df["btc_mining_payoff_1"] = 1 / (df["btc_block_rate"] * df["btc_total_payoff"])
    df["btc_mining_payoff_2"] = 1 / df["btc_total_payoff"]

    # Previous high mining payoffs
    df["eth_previous_mining_payoff_1"] = (
        df["eth_mining_payoff_1"].cummax() / df["eth_mining_payoff_1"]
    )
    df["eth_previous_mining_payoff_2"] = (
        df["eth_mining_payoff_2"].cummax() / df["eth_mining_payoff_2"]
    )

    df["btc_previous_mining_payoff_1"] = (
        df["btc_mining_payoff_1"].cummax() / df["btc_mining_payoff_1"]
    )
    df["btc_previous_mining_payoff_2"] = (
        df["btc_mining_payoff_2"].cummax() / df["btc_mining_payoff_2"]
    )

    return df


def econ_hash_rate(df):
    """
    :df: df_basic
    """

    # ETH
    """ hashrate = difficulty / block_time
            avg_difficulty = eth_difficulty_sum / eth_n_blcoks
            avg_block_time = 3600 / eth_n_blocks (estimate over last hour)
        
        => hashrate = (eth_difficulty_sum / eth_n_blocks) * (eth_n_blocks / 3600)
                    = eth_difficulty_sum / 3600
    """
    df["eth_hashrate"] = df["eth_difficulty_sum"].astype(float) / 3600

    # BTC
    df["btc_blocks_found_last_24_hours"] = df["btc_n_blocks"].rolling(24).sum()

    for index, row in df.iterrows():
        btc_daily_hashrate = daily_hash_rate_btc(
            difficulty=row["btc_difficulty"],
            blocks_found=row["btc_blocks_found_last_24_hours"],
        )
        df.at[index, "btc_daily_hashrate"] = btc_daily_hashrate

    # Previous high hash rate
    df["eth_previous_hashrate"] = df["eth_hashrate"].cummax() / df["eth_hashrate"]
    df["btc_previous_hashrate"] = (
        df["btc_daily_hashrate"].cummax() / df["btc_daily_hashrate"]
    )
    return df

def econ_excess_block_space(df):
    """
    :df: df_basic
    """
    # ETH
    # Note: replacing this fn with just congestion factors, but could revisit
    # TODO: verify that eth_gaslimit and eth_gasused are at the same scale
    df["eth_excess_block_space"] = df["eth_gaslimit_sum"] - df["eth_gasused_sum"]

    # BTC
    # TODO: think this calculation has to be changed
    df["btc_excess_block_space"] = df["btc_block_size"] / df["btc_n_transactions"]

    return df


def econ_social_factors(df):
    """
    :df: df_basic
    """
    # ETH
    # TODO: do we want log(total gas for hour) or log(gas_block1) + log(gas_block2) + ...?
    # decided to take average gasused / block for now
    # might want to be comparable to BTC units though, but gas not same as bytes anyway
    df["eth_social_value"] = (-1) * np.log(df["eth_gasused_sum"] / df["eth_n_blocks"])
    df["eth_social_cost"] = 1 / (df["eth_gasused_sum"] / df["eth_n_blocks"])

    # BTC
    # Note: btc_block_size is mean variable over hour
    df["btc_social_value"] = (-1) * np.log(df["btc_block_size"])
    df["btc_social_cost"] = 1 / df["btc_block_size"]

    return df


def econ_computational_burden(df):
    """
    :df: df_basic
    """
    # TODO: same questions as previous function
    # ETH
    df["eth_computational_burden"] = (df["eth_blocksize_sum"] / df["eth_n_blocks"]) * (
        np.log(df["eth_blocksize_sum"] / df["eth_n_blocks"])
    ) ** 2

    # BTC
    df["btc_computational_burden"] = (
        df["btc_block_size"] * (np.log(df["btc_block_size"])) ** 2
    )

    return df


def econ_congestion_factors(df):
    """
    :df: df_basic
    """
    # ETH
    df["eth_congestion_1"] = df["eth_gasused_sum"] / df["eth_gaslimit_sum"]
    df["eth_congestion_2"] = df["eth_congestion_1"] ** 2

    df["eth_congestion_3"] = df["eth_congestion_1"].apply(
        lambda x: 1 if x > settings.CONGESTION_CUTOFF else 0
    )

    # BTC
    df["btc_block_limit"] = df["btc_block_size"].cummax()
    df["btc_congestion_1"] = df["btc_block_size"] / df["btc_block_limit"]
    df["btc_congestion_2"] = df["btc_congestion_1"] ** 2

    df["btc_congestion_3"] = df["btc_congestion_1"].apply(
        lambda x: 1 if x > settings.CONGESTION_CUTOFF else 0
    )

    return df


def econ_poisson_proxy(df):
    """
    :df: df_basic
    """
    # TODO: as above (in mining payoff terms), might want longer time period to estimate rate
    # ETH
    df["eth_mu"] = 1 / df["eth_n_blocks"]

    # BTC
    df["btc_mu"] = 1 / df["btc_n_blocks"]

    return df


def econ_congestion_pricing(df):
    """
    :df: df_basic
    """
    # TODO: Think about whether congestion_pricing_1 should use a different numerator.
    # ETH
    # Note: eth_weighted_average_gasprice = eth_weighted_gasprice_sum
    df["eth_congestion_pricing_1"] = df["eth_congestion_1"] / (
        df["eth_weighted_gasprice_sum"]
    )
    df["eth_congestion_pricing_2"] = df["eth_gaslimit_sum"] / (
        df["eth_weighted_gasprice_sum"] * df["eth_gasused_sum"]
    )
    df["eth_congestion_pricing_3"] = (df["eth_gaslimit_sum"]) ** 2 / (
        df["eth_weighted_gasprice_sum"] * df["eth_gasused_sum"]
    )

    # BTC
    df["btc_weighted_average_tx_fee"] = df["btc_fee_unit_sum"] / df["btc_n_blocks"]
    df["btc_congestion_pricing_1"] = (
        df["btc_congestion_1"] / df["btc_weighted_average_tx_fee"]
    )
    df["btc_congestion_pricing_2"] = df["btc_block_limit"] / (
        df["btc_fee_sum"] / df["btc_n_blocks"]
    )
    df["btc_congestion_pricing_3"] = (df["btc_block_limit"]) ** 2 / (
        df["btc_fee_sum"] / df["btc_n_blocks"]
    )

    return df


def econ_addresses(df):
    """
    :df: df_basic
    """
    # ETH
    df["eth_unique_addresses"] = df["eth_n_unique_from_address"]
    df["eth_spreading"] = (
        df["eth_n_unique_to_address"] / df["eth_n_unique_from_address"]
    )

    # BTC
    df["btc_unique_addresses"] = df["btc_n_unique_inptput_address"]
    df["btc_spreading"] = (
        df["btc_n_unique_output_address"] / df["btc_n_unique_inptput_address"]
    )

    return df


def transform_economic(df):
    print("Making dataset with economic variables...")
    df_economic = econ_mining_payoff_factors(df)
    df_economic = econ_addresses(df_economic)
    df_economic = econ_computational_burden(df_economic)
    # df_economic = econ_excess_block_space(df_economic) #note: replacing with congestion factors, could revisit
    df_economic = econ_congestion_factors(df_economic)
    df_economic = econ_congestion_pricing(df_economic)
    df_economic = econ_hash_rate(df_economic)
    df_economic = econ_poisson_proxy(df_economic)
    df_economic = econ_social_factors(df_economic)

    return df_economic


def transform_btc_eth_final(df_0):
    print("transforming btc and eth data...")
    df_1 = df_0.copy().astype("float")

    df_1["eth_spread_proxy"] = (df_1.eth_high - df_1.eth_low) / df_1.eth_close
    df_1["eth_difficulty_pb"] = df_1.eth_difficulty_sum / df_1.eth_n_blocks
    df_1["eth_n_fr_address_pb"] = df_1.eth_n_unique_from_address / df_1.eth_n_blocks
    df_1["eth_n_to_address_pb"] = df_1.eth_n_unique_to_address / df_1.eth_n_blocks
    df_1["eth_blocksize_pb"] = df_1.eth_blocksize_sum / df_1.eth_n_blocks
    df_1["eth_n_tx_pb"] = df_1.eth_n_transactions / df_1.eth_n_blocks
    # df_1["eth_ethersupply_growth"] = (
    #     df_1.eth_ethersupply_sum - df_1.eth_ethersupply_sum.shift(1)
    # ) / df_1.eth_ethersupply_sum.shift(1)
    
    df_1["eth_gaslimit_pb"] = df_1.eth_gaslimit_sum / df_1.eth_n_blocks
    df_1["eth_gasused_pb"] = df_1.eth_gasused_sum / df_1.eth_n_transactions
    df_1["eth_gasused_squared_pt"] = df_1.eth_gasused_square_sum / df_1.eth_n_transactions
    df_1["eth_gasprice_pt"] = df_1.eth_gasprice_sum / df_1.eth_n_transactions
    df_1["eth_gasprice_squared_pt"] = df_1.eth_gasprice_square_sum / df_1.eth_n_transactions
    df_1["eth_weighted_gasprice"] = df_1.eth_weighted_gasprice_sum  
    df_1["eth_weighted_squared_gasprice"] = df_1.eth_weighted_gasprice_square_sum


    df_1["btc_spread_proxy"]    = (df_1.btc_high - df_1.btc_low) / df_1.btc_close
    df_1["btc_n_tx_pb"]         = df_1.btc_n_transactions / df_1.btc_n_blocks
    df_1["btc_miner_reward_pb"] = df_1.btc_miner_reward / df_1.btc_n_blocks
    df_1["btc_n_fr_address_pb"] = df_1.btc_n_unique_inptput_address / df_1.btc_n_blocks
    df_1["btc_n_to_address_pb"] = df_1.btc_n_unique_output_address / df_1.btc_n_blocks
    
    df_1["btc_fee_unit_pb"]     = df_1.btc_fee_unit_sum / df_1.btc_n_transactions
    df_1["btc_fee_unit_weighted"] = df_1.btc_fee_unit_weighted_sum  
    df_1["btc_fee_unit_square_weightd"] = df_1.btc_fee_unit_square_weightd_sum
    
    df_2 = df_1
    #[col_selected]
    return df_2
    

def transform_uni_final(df_0):
    print("transforming uniswap data...")
    df_1 = df_0.copy().astype("float")
    df_1["uniswap2_usdt_combinedBalanceInEth"] = (
        df_1.eth_usdt_mints_eth_amount.cumsum()
        - df_1.eth_usdt_burns_eth_amount.cumsum()
        + df_1.eth_usdt_swaps_eth_amount.cumsum()
    )
    df_1["uniswap2_usdc_combinedBalanceInEth"] = (
        df_1.usdc_eth_mints_eth_amount.cumsum()
        - df_1.usdc_eth_burns_eth_amount.cumsum()
        - df_1.usdc_eth_swaps_eth_amount.cumsum()
    )
    df_1["uniswap2_dai_combinedBalanceInEth"] = (
        df_1.dai_eth_mints_eth_amount.cumsum()
        - df_1.dai_eth_burns_eth_amount.cumsum()
        - df_1.dai_eth_swaps_eth_amount.cumsum()
    )

    df_1["uniswap1_combinedBalanceInEth"] = (
        df_1.USDC_combinedBalanceInEth
        + df_1.SAI_combinedBalanceInEth
        + df_1.Dai_combinedBalanceInEth
    )

    df_1["uniswap_combinedBalanceEth"] = (
        df_1["uniswap1_combinedBalanceInEth"]
        + df_1["uniswap2_dai_combinedBalanceInEth"]
        + df_1["uniswap2_usdc_combinedBalanceInEth"]
        + df_1["uniswap2_usdt_combinedBalanceInEth"]
    )

    df_1["uniswap_RemoveLiquidityEth"] = (
        df_1.dai_eth_burns_eth_amount
        + df_1.usdc_eth_burns_eth_amount
        + df_1.eth_usdt_burns_eth_amount
    ) * 2 + df_1.rl_ethAmount

    df_1["uniswap_AddLiquidityEth"] = (
        df_1.dai_eth_mints_eth_amount
        + df_1.usdc_eth_mints_eth_amount
        + df_1.eth_usdt_mints_eth_amount
    ) * 2 + df_1.al_ethAmount

    df_1["uniswap_inflow"] = (
        df_1["uniswap_AddLiquidityEth"] - df_1["uniswap_RemoveLiquidityEth"]
    ) / df_1["uniswap_combinedBalanceEth"]

    df_1["uniswap_TradeAmountEth"] = (
        df_1.Dai_tradeVolumeEth
        + df_1.USDC_tradeVolumeEth
        + df_1.SAI_tradeVolumeEth
        + +df_1.eth_usdt_swaps_eth_amount
        + df_1.usdc_eth_swaps_eth_amount
        + df_1.dai_eth_swaps_eth_amount
    )

    df_1["uniswap_tradeflow"] = (
        df_1["uniswap_TradeAmountEth"] / df_1["uniswap_combinedBalanceEth"]
    )

    df_1["uniswap_TradeAmountEth_ema"] = (
        df_1["uniswap_TradeAmountEth"].ewm(alpha=0.1, ignore_na=True).mean()
    )

    df_1["uniswap_TradeAmountEth_lag"] = df_1.uniswap_TradeAmountEth_ema.shift(1)

    df_1["uniswap_tradeAmountEth_growth"] = (
        df_1.uniswap_TradeAmountEth - df_1.uniswap_TradeAmountEth.shift(1)
    ) / df_1.uniswap_TradeAmountEth.shift(1)

    col_selected=[
        "uniswap_inflow",
        "uniswap_tradeflow",
        "uniswap_tradeAmountEth_growth",
    ]
    df_2 = df_1[col_selected]
    return df_2
    

def data_factory():
    print("Making df_basic...")
    eth_data = preprocess_eth_data()
    btc_data = preprocess_btc_data()
    uniswapv1_data = preprocess_uniswapv1_data()
    uniswapv1_events = preprocess_uniswapv1_events()
    uniswapv2_data = preprocess_uniswapv2_data()

    df_basic = pd.concat(
        [eth_data, btc_data, uniswapv1_data, uniswapv1_events, uniswapv2_data],
        sort=False,
        axis=1,
        join="outer",
    )
    df_basic.to_pickle("final/df_basic.pk")
    

    df_btc_eth_final = transform_btc_eth_final(df_basic)
    
    df_uni_final = transform_uni_final(df_basic)
    
    df_final=pd.concat([df_btc_eth_final, df_uni_final],
                           sort=False,
                           axis=1,
                           join="outer",
                       )
    df_final.to_pickle("final/df_final.pk")
    df_economic = transform_economic(df_basic)
    df_economic.to_pickle("final/df_economic.pk")

    print("Building datasets succeeded!")


if __name__ == "__main__":
    data_factory()