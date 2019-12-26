#--coding:utf-8--

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
import warnings
import os
import glob
import pdb
warnings.filterwarnings('ignore')


dataOutput = pd.DataFrame(index=['NumOfSamples','KNN','LogisticRegression','SVM(rbf)','RandomForest'])

dir_path = os.path.dirname(os.path.realpath(__file__))
for f in glob.glob(dir_path + '\*.xlsx'):
    data = pd.read_excel(f, header=0, encoding='latin-1')
    stock = f.split('$')[1]
    stock = stock.split('.')[0]

    x = np.array(data[['WeightedPolarity_scaled']])[:-1]
    y = np.array(data['BuyOrSell'])[:-1]

    # KNN - K-Nearest-Neighbors
    neigh = KNeighborsClassifier(n_neighbors=5)
    KNN_score = cross_val_score(neigh, x, y, cv=TimeSeriesSplit(n_splits=10).split(data.iloc[:-1, :])).mean()

    # Logistic Regression
    logreg = LogisticRegression(fit_intercept=True)
    Logistic_score = cross_val_score(logreg, x, y, cv=TimeSeriesSplit(n_splits=10).split(data.iloc[:-1, :])).mean()

    # Support Vector Machines (SVM) with rbf kernel
    svmRbf = SVC(kernel='rbf')
    SVM_score = cross_val_score(svmRbf, x, y, cv=TimeSeriesSplit(n_splits=10).split(data.iloc[:-1, :])).mean()

    # Random Forest
    forest_reg = RandomForestClassifier()
    RF_score = cross_val_score(forest_reg, x, y, cv=TimeSeriesSplit(n_splits=10).split(data.iloc[:-1, :])).mean()

    dataOutput[stock] = [len(y),KNN_score,Logistic_score,SVM_score,RF_score]

dataOutput = dataOutput.transpose()
dataOutput.to_excel("ClassificationResults_5 fold.xlsx")







