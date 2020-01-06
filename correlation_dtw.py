#--coding:utf-8--

from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import heapq
import os
import glob

import pdb


def normalization(series):
    getSeries = (series- np.median(series))/np.std(series)
    return getSeries

def dist_for_float(series, tseries):
    dist = pow(abs(series - tseries),2)
    return dist

#dynamic time warping
def dtw(series, tseries, dist_func):
    mat = [([[0, 0, 0, 0] for j in range(len(series))]) for i in range(len(tseries))]
    for x in range(len(series)):
        for y in range(len(tseries)):
            dist = dist_func(series[x], tseries[y])
            mat[y][x] = [dist, 0, 0, 0]
    elem_0_0 = mat[0][0]
    elem_0_0[1] = elem_0_0[0]
    for x in range(1, len(series)):
        mat[0][x][1] = mat[0][x][0] + mat[0][x - 1][1]
        mat[0][x][2] = x - 1
        mat[0][x][3] = 0
    for y in range(1, len(tseries)):
        mat[y][0][1] = mat[y][0][0] + mat[y - 1][0][1]
        mat[y][0][2] = 0
        mat[y][0][3] = y - 1

    for y in range(1, len(tseries)):
        for x in range(1, len(series)):
            distlist = [mat[y][x - 1][1], mat[y - 1][x][1], mat[y - 1][x - 1][1]]
            mindist = min(distlist)
            idx = distlist.index(mindist)
            mat[y][x][1] = mat[y][x][0] + mindist
            if idx == 0:
                mat[y][x][2] = x - 1
                mat[y][x][3] = y
            elif idx == 1:
                mat[y][x][2] = x
                mat[y][x][3] = y - 1
            else:
                mat[y][x][2] = x - 1
                mat[y][x][3] = y - 1

    result = mat[len(tseries) - 1][len(series) - 1]
    retval = result[1]
    path = [(len(series) - 1, len(tseries) - 1)]
    while True:
        x = result[2]
        y = result[3]
        path.append((x, y))

        result = mat[y][x]
        if x == 0 and y == 0:
            break
    return retval, sorted(path)


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataOutput = pd.DataFrame()
    for f in glob.glob(dir_path + '\*.xlsx'):
        data = pd.read_excel(f, index_col=0, header=0, encoding='latin-1')
        stock = f.split('$')[1]
        stock = stock.split('.')[0]

        '''
        dist = []
        for i in range(1,6):
            tseries = np.array(data['WeightedPolarity_scaled'][:-i])
            series = np.array(data['PctChange_scaled'].shift(-i)[:-i])
            
            #dist_for_series = np.sum((tseries - series) ** 2)/len(tseries)
            value, path = dtw(series, tseries, dist_for_float)
            dist.append(dist_for_series)
        
        min_num_index = list(map(dist.index, heapq.nsmallest(10, dist)))
        dataOutput[stock] = min_num_index
        '''
        tseries = data['WeightedPolarity_scaled'][:-1]
        series = data['PctChange_scaled'].shift(-1)[:-1]
        if len(tseries) < 75 or len(series) < 75:
            continue
        else:
            tseries = tseries[0:75]
            series = series[0:75]
            #pdb.set_trace()
            #dist_for_series = np.sum((tseries - series) ** 2)/len(tseries)
            value, path = dtw(series, tseries, dist_for_float)
            dataOutput[stock] = [value]

    dataOutput.to_excel('correlation_dtw_75.xlsx')




