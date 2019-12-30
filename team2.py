#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:28:56 2019

@author: helgiloga
"""

import team2Functions as fun
import pandas_datareader.data as web

team1Data = fun.simulateData()
stocks = team1Data.Ticker.unique()

for stock in stocks:
    data = team1Data[team1Data['Ticker'] == stock]
    feature_cols = ['Pos', 'Neg', 'Neu']
    
    start = '2018-01-01'
    end = '2019-05-20'
    
    ts = web.DataReader(stock,'yahoo',start, end)
    ts = ts.Close
    
    lag = 20
    y_bin, y_ret = [],[]
    
    for i in range(len(ts)-lag):
        if ts[i+lag] > ts[i]:
            y_bin.append(1)
        else:
            y_bin.append(0)
        y_ret.append((ts[i+lag]-ts[i])/ts[i])
            
    data = data[:len(y_bin)]
    data['response'] = y_bin
    
    fun.classify(data)
    
    data['response'] = y_ret
    
    #fun.regress(data)
    






