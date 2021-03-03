import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, MinMaxScaler
from functools import reduce


class MinMaxScalerQ(TransformerMixin):
    #MinMaxScaler but for pandas DataFrames
    #Set the min and max cuantile variables and drop those variables being lower than the low limit and greather than upper limit
    
    def __init__(self, X, columns = None, q_min = 0.25, q_max = 0.75):
        self.columns = columns
        if self.columns == None:
            self.columns = X.select_dtypes(include=[np.number]).columns
        self.q_min = q_min
        self.q_max = q_max
        self.mms = None
        self.quantile = None
        
    def fit(self, X, y=None):
        self.mms = MinMaxScaler()
        self.quantile = pd.DataFrame.quantile(X,[self.q_min,self.q_max], axis = 0)
        self.quantiledict = self.quantile.rename(index={self.q_min: 'min', self.q_max: 'max'}).to_dict()
        for i in self.columns:
            X[i] = X[i].apply(lambda y: np.nan if (y < self.quantiledict[i]['min']) | (y > self.quantiledict[i]['max']) else y)
        X = X[self.columns]
        self.mms.fit(X)
        return self
    
    def transform(self, X):
        # assumes X is a DataFrame
        Xmms_columns = self.columns
        X = X[Xmms_columns]
        Xmms = self.mms.transform(X)
        Xscaled = pd.DataFrame(Xmms, index=X.index, columns=X.columns)
        for i in Xscaled.columns:
            Xscaled[i] = Xscaled[i].apply(lambda y: 0 if (y < 0) else (1 if (y > 1) else y))
        return Xscaled