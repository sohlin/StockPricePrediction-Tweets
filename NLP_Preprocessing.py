#--coding:utf-8--

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas_datareader.data as web
import xlsxwriter
import os
import glob
from textblob import TextBlob


dir_path = os.path.dirname(os.path.realpath(__file__))
for f in glob.glob(dir_path + '\*.xlsx'):
    data = pd.read_excel(f,header = 0,encoding='latin-1', sheet_name = "Stream")

    # get tweets
    stock = f.split('_')[3].upper()
    Tweet = data['Tweet content']

    # sentiment analysis: polarity & subjectivity
    polarity = Tweet.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    subjectivity = Tweet.apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    # Merge the sentiment with the origin data
    data['Polarity'] = polarity
    data['Subjectivity'] = subjectivity
    data_tweets = data[(data['Date'] >= '2016-03-10') & (data['Date'] <= '2016-06-15')]

    # Adding a datetime index
    data_tweets['datetime'] = pd.to_datetime(data_tweets['Date'])
    dataTweets = data_tweets.set_index('datetime')

    dataTweets = dataTweets[['Hour', 'Tweet content', 'Favs', 'RTs', 'Followers', 'Following', 'Is a RT',
                             'Hashtags', 'Symbols', 'Polarity', 'Subjectivity']]

    # Drop neutral sentiment & Perform weighted polarity
    dataTweets = dataTweets[(dataTweets[['Polarity']] != 0).all(axis=1)]
    dataTweets = dataTweets[np.isfinite(dataTweets['Followers'])]
    dataTweets['WeightedPolarity'] = dataTweets['Polarity'] * dataTweets['Followers']

    # Scale the column "WeightedPolarity"
    weitP = dataTweets[['WeightedPolarity']].values.astype(float)
    scaler = StandardScaler().fit(weitP)
    scaledData = scaler.transform(weitP)
    dataTweets['WeightedPolarity_scaled'] = scaledData

    # Create daily MEANS of each column
    dailyMean = (dataTweets.groupby(dataTweets.index).mean())
    dailyMean.sort_index(inplace=True)

    # stock data from Yahoo! Finance
    startDate = dailyMean.index[0]
    endDate = dailyMean.index[-1]  # dt.datetime.now()
    stockData = web.DataReader(stock, 'yahoo', startDate, endDate)

    stockData.columns = ['High', 'Low', 'Open', 'Close', 'Volume_stock', 'Adj_Close_stock']
    stockData = stockData.reindex(index=dailyMean.index)

    # Interpolate for missing weekend stock data
    stockData[['High', 'Low', 'Open', 'Close', "Volume_stock", "Adj_Close_stock"]] = \
        stockData[['High', 'Low', 'Open', 'Close', "Volume_stock", "Adj_Close_stock"]] \
            .interpolate(method='linear', limit_direction='forward', axis=0)

    stockData['HLVolatility'] = (stockData['High'] - stockData['Low']) / stockData['Adj_Close_stock'] * 100.0
    stockData['PctChange'] = (stockData['Close'] - stockData['Open']) / stockData['Open'] * 100.0

    # Add column for daily percent change scaled - stock
    stockPerc = stockData[['PctChange']].values.astype(float)
    scaler = StandardScaler().fit(stockPerc)
    scaledData = scaler.transform(stockPerc)
    stockData['PctChange_scaled'] = scaledData

    # Combine the tweet sentiment data with the stock data
    fullData = pd.concat(
        [stockData[['Volume_stock', 'Adj_Close_stock', 'HLVolatility', 'PctChange', 'PctChange_scaled']], \
         dailyMean], axis=1, sort=False)

    # Data Description
    dataDescription = pd.DataFrame.describe(fullData)

    # get tommorow's pctChange as tag
    fullData['PredictedChange'] = fullData['PctChange'].shift(-1)

    # Produce signal, 1: buy; 0: sell or no signal
    fullData['BuyOrSell'] = fullData['PctChange'] > 0
    fullData['BuyOrSell'] = fullData['BuyOrSell'].shift(-1)

    '''
    # Plot the sentiment of each day
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] =8.0
    fig_size[1] = 4.0
    x = fullData['WeightedPolarity_scaled']
    plt.hist(x, normed=True, bins=250)
    plt.xlabel('WeightedPolarity_scaled')
    plt.ylabel('Number')
    plt.title(stock)
    plt.show()
    '''

    # Save result to file
    fullData.to_excel('fullData.xlsx')
    os.rename('fullData.xlsx', '$' + stock + '.xlsx')

