import pandas as pd
from .base import BasePipeStep
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class SelectColumns(BasePipeStep):

    def __init__(self, columns):
        self.columns = columns
    
    def transform(self, X):
        X = X.copy()
        return X[self.columns]

class OneHotEncoderDf(SelectColumns):

    def __init__(self, columns = [], fillna_values='NULL'):
        self.columns = columns
        self.fillna_values= fillna_values
    
    def fit(self, X, y=None):
        self.one_hot = OneHotEncoder(handle_unknown='ignore')
        self.one_hot.fit(X[self.columns].fillna(self.fillna_values))
        return self
    
    def transform(self, X):
        X = X.copy()[self.columns].fillna(self.fillna_values)        
        return pd.DataFrame(
            self.one_hot.transform(X).toarray(),
            columns = self.one_hot.get_feature_names(self.columns),
            index = X.index
        )

class FillNumericData(BasePipeStep):
    imputer = SimpleImputer()

    def __init__(self, columns, impute_strategy='mean', **kwargs):
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.columns = columns
        self.impute_strategy = impute_strategy
        

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns], y)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns = X.columns, index=X.index)


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