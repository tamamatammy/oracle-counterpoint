# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:05:37 2021

@author: yang8
"""

import multiprocessing as mp
from multiprocessing import Process
import extract_celo as c
import time as time
import pandas as pd

print("Number of processors: ", mp.cpu_count())
dir_op="Output/New/sample/"

def block_func():
    start=1424320
    end=4972900 #or request_latest_block()
    c.request_blocks_ts(start, end, save=True, filename="celo_blocks_ts_raw")

    
def transfer_func():
    latest_cursor="YXJyYXljb25uZWN0aW9uOjEzMzk4MDA="
    c.request_celo_transfers_ts(latest_cursor, save=True, filename="celo_transfers_ts_raw")
    

########################################################################################
# multiprocessing #

start=time.time()

if __name__=='__main__':
    p1 = Process(target=block_func)

    p2 = Process(target=transfer_func)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    
end=time.time()
runtime_secondes=end - start
print('time series request time taken:', time.strftime("%H:%M:%S",time.gmtime(runtime_secondes)))

########################################################################################

# backfill data #

#c.backfill_request_error("celo_blocks_ts_raw")

#c.backfill_request_error("celo_transfers_ts_raw")

########################################################################################
# threading #

# from threading import Thread
# start=time.time()

# if __name__ == '__main__':
#     t1=Thread(target = func_block)
#     t1.setDaemon(True)
#     t1.start()
#     t2=Thread(target = func_transfers)
#     t2.setDaemon(True)
#     t2.start()
    
# end=time.time()
# runtime_secondes=end - start
# print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(runtime_secondes)))

    
# if t1.IsAlive():
#     t1._Thread__stop() 
# if t2.IsAlive():
#     t2._Thread__stop() 
    