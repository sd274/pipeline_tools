import pandas as pd
from .base import BasePipeStep
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class SelectColumns(BasePipeStep):

    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X):
        X = X.copy()
        return X[self.columns]

class OneHotEncoderDf(SelectColumns):
    
    def fit(self, X, y=None):
        self.one_hot = OneHotEncoder(handle_unknown='error', drop='first')
        self.one_hot.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()[self.columns]        
        return pd.DataFrame(
            self.one_hot.transform(X).toarray(),
            columns = self.one_hot.get_feature_names(self.columns)
        )


class ScaleNumeric(SelectColumns):
    
    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.columns])
        return self
        
    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

class ToNumeric(SelectColumns):
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X