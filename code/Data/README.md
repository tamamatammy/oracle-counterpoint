# Data Sources
* Google Cloud Big Query Data Warehouse
* Coinbase Pro API
* Bitcoinity data - https://data.bitcoinity.org/markets/volume/30d?c=e&t=b
* Celo data - https://explorer.celo.org/blocks

# Raw Data Description
This section provides descrptions of the raw data extracted for various data sources
## BTC Block Data Description
BTC data from Google Cloud BigQuery datbase were collected at block level
* btc_fee_sum: total transaction fee per block
* btc_fee_unit_sum: total unit fee (transaction fee/transaction size) per block
* btc_fee_unit_square_sum:  total squared unit fee to hourly level per
* btc_fee_unit_weighted_sum: total weighted unit fee (trasaction size/block size*(fee/tranasaction size)) per block
* btc_miner_reward: miner reward per block
* btc_n_transactions: number of transactions per block
* btc_n_unique_inptput_address: number of sender address per block
* btc_n_unique_output_address: number of receipt address per block
* btc_block_size: block size
* btc_block_bits: block bits
* btc_block_nonce: block nonce

## Ethereum Block Data Description
ETH data from Google Cloud BigQuery database were aggregated to hourly level
###  Block Data
* eth_n_blocks: number of blocks per hours
* eth_diffiulty_sum: summation of block difficulty to hourly level
* eth_gaslimit_sum: summation of gas limit to hourly level
* eth_gasused_sum: summation of gas used to hourly level
* eth_gasused_square_sum: summation of squared gas used to hourly level
* eth_blocksize_sum: summation of block size to hourly level
* eth_ethersupply_sum: total ether supply (close of the hour); accumulated sum of supply changes up to the hour (from genesis and reward txs in bigquery)

### Transaction Data
* eth_gasprice_sum: sumation of gas price per transaction to hourly level
* eth_gasprice_square_sum: summation of squred gas price per transaction to hourly level
* eth_weighted_gasprice_sum: sum of weighted gas price (per tx: receipt_gas_used/gas_used (in hour) * gas_price) = weighted avg gas price in the hour
* eth_weighted_gasprice_square_sum: summation of weighted squared gas price (receipt_gas_used/gas_used * gas_price^2)
* eth_n_unique_from_address: number of from_address per hour
* eth_n_transactions: number of transactions per hour

# Development Sample Dataset Description
The following section provides descriptions of the data and its transformation (if there are any) to construct the final development sample dataset for model development

## Selected ETH Block Data and Transformation
* eth_close
* (eth_high - eth_low) / eth_close
* eth_volumn
* eth_n_blocks
* eth_difficulty_sum / eth_n_blocks
* eth_gaslimit_sum / eth_n_blocks
* eth_gasused_sum  / eth_n_blocks
* eth_blocksize_sum / eth_n_blocks
* Change in eth_ethersupply_sum (% or abs)
* eth_n_transactions / eth_n_blocks (note: to avoid correlation with number of blocks)
* eth_gasprice_sum / eth_n_transactions
* eth_weighted_gasprice_sum / eth_n_transactions
* eth_n_unique_from_address / eth_n_blocks
* eth_n_unique_to_address / eth_n_blocks
* maybe for volatility
** eth_gasused_square_sum 
** eth_gasprice_square_sum

## Selected BTC Block Data and Transformation
* btc_close
* (btc_high - btc_low)/ btc_close
* btc_volumn
* btc_n_blocks (check whether exists)
* btc_n_transactions / btc_n_blocks
* btc_fee_unit_sum / btc_n_transactions
* btc_miner_reward / btc_n_blocks
* btc_n_unique_inptput_address / btc_n_blocks
* btc_n_unique_output_address / btc_n_blocks
* btc_block_size (note: already transformed as mean block size)
* btc_difficulty (note: already transformed as mean difficulty)

** maybe
* btc_fee_sum / btc_n_transactions (note: unit fee might be better estimatior)
* btc_fee_unit_weighted_sum (note: volatility)
* btc_fee_unit_square_weightd_sum

## Selected Uniswap Data and Transformation
* (sum of burns in eth amount (v2) + sum of remove liquidity in eth (v1?))- (sum of mints in eth amount (v2) + sum of add liquidity in eth (v1?))/ combined balance (v1 + v2(?))
* (exchange trade volumn (v1) + swaps eth amount (v2)) / combined balance (v1 + v2(?))
* growth of (exchange trade volumn (v1) + swaps eth amount (v2))


    
