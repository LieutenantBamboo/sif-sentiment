#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:28:56 2019

@author: helgiloga
"""
import pandas as pd
import numpy as np
import team2Functions as fun
import pandas_datareader.data as web
#'teeam1Data' is the sentiment data
#'stocks' are unique stock ticker names we have
team1Data = fun.simulateData()
stocks = team1Data.Ticker.unique()
pred=np.array([])

for stock in stocks:
    data = team1Data[team1Data['Ticker'] == stock]
    feature_cols = ['Pos', 'Neg', 'Neu']
    
    start = '2018-01-01'
    end = '2019-05-20'
#'ts': stock price
    ts = web.DataReader(stock,'yahoo',start, end)
    ts = ts.Close
    
    lag = 20
    y_bin, y_ret = [],[]
#'y_bin': binary return
#'y_ret': return 
    for i in range(len(ts)-lag):
        if ts[i+lag] > ts[i]:
            y_bin.append(1)
        else:
            y_bin.append(0)
        y_ret.append((ts[i+lag]-ts[i])/ts[i])
            
    data = data[:len(y_bin)]
    data['response'] = y_bin
    
    y_pred,y_prob=fun.classify(data)
    pred=np.concatenate((pred,y_pred),axis=None)
    y_prob=y_prob.sort_index(axis=0)
    if stock==stocks[0]:
        prob=y_prob
    else:
        prob=pd.concat([prob,y_prob])
    #data['response'] = y_ret
    
    #fun.regress(data)
   



