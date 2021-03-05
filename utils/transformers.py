import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from functools import reduce

class DummyTransformer(TransformerMixin):
    def __init__(self, columns):
        self.dict_values = None
        self.columns = columns

    def fit(self, X, y=None):
        # assumes all columns of X are categorical
        cols = self.columns
        dict_col_vals = dict()
        for col in cols:
            a,_ = np.unique(X[col].astype(str).dropna(), return_inverse=True)
            dict_col_vals[col]=a
        self.dict_values = dict_col_vals    
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        cols_d = set(self.dict_values.keys())
        cold_x = set(X.columns)
        cols = list(cols_d.intersection(cold_x))
        Xdum = pd.DataFrame()
        for col in cols:
            X[col] = X[col].apply(lambda x: x if x in self.dict_values[col] else np.nan)
            s_values = pd.Series(self.dict_values[col])
            Xdum_col = pd.concat([s_values,X[col]])
            Xdum_col = pd.get_dummies(Xdum_col)
            Xdum_col = Xdum_col.iloc[len(s_values):,:]
            Xdum_col.columns = [col + '_' + col_d for col_d in  Xdum_col.columns]
            Xdum = pd.concat([Xdum,Xdum_col],axis=1)
        return Xdum

    
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X, *_):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X[self.columns])
        else:
            raise TypeError("Este Transformador solo funciona en DF de Pandas")
    
    def fit(self, X, *_):
        return self

    
    
class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames
    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled    
    
    
    
class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


    
class DFKBinsOrdinalDiscretizer(TransformerMixin):
    def __init__(self,cols , n_bins=5,  strategy='quantile'):
        self.n_bins = n_bins
        self.encode = 'ordinal'
        self.strategy = strategy
        self.cols = cols

    def fit(self, X, y=None):
        self.kbd = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode,strategy=self.strategy)
        self.kbd.fit(X[self.cols])
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xkbd = self.kbd.transform(X[self.cols])
        Xkbd_df = pd.DataFrame(Xkbd, index=X.index, columns=X[self.cols].columns)
        return Xkbd_df    
    
    

class FillNaDict( BaseEstimator, TransformerMixin ):
    def __init__( self, col_fill_dict, final = False ):
        self._col_fill_dict = col_fill_dict
        self._final = final
    
    def fit( self, X, y = None ):
        # if value equals mode, mean, or median, these measures need to be calculated 
        l_mode = [k for k, v in self._col_fill_dict.items() if v == 'mode']
        l_mean = [k for k, v in self._col_fill_dict.items() if v == 'mean']
        l_median = [k for k, v in self._col_fill_dict.items() if v == 'median']

        d_mode = X[l_mode].mode().to_dict()
        d_mean = X[l_mean].mean().to_dict()
        d_median = X[l_median].median().to_dict()

        for k,v in d_mode.items():
            self._col_fill_dict[k] = v[0]
            
        for k,v in d_mean.items():
            self._col_fill_dict[k] = v
            
        for k,v in d_median.items():
            self._col_fill_dict[k] = v

        return self 
    

    def transform( self, X, y = None ):
        if((self._final) and (not(all(elem in col_fill_dict.keys())  for elem in X.columns))):
            raise ValueError("""Missing column in dict""")
        return X.fillna(self._col_fill_dict)
    

    
class DFOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.dict_enc = {}
    
    def fit(self, X, *_):
        X = X[self.columns]
        for col in self.columns:
            self.dict_enc[col] = sorted(list(X[col].unique()))
        return self
    
    def transform(self, X, *_):
        X = X[self.columns].copy()
        for col in self.columns:
            l_vals = self.dict_enc[col]
            X.loc[:,col] = X[col].map({k:v for v,k in list(enumerate(l_vals))})
            X.loc[:,col] = X.loc[:,col].fillna(len(l_vals))
        return X
    
    def inverse_transform(self, X):
        X = X[self.columns].copy()
        for col in self.columns:
            l_vals = self.dict_enc[col]
            X.loc[:,col] = X[col].map({k:v for k,v in list(enumerate(l_vals))})
        return X