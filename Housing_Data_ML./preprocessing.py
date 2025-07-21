

import pandas as pd 
import numpy as np
import matplotlib as mp

# Create 'Pipes' in the pipeline using scikit pipeline. We begin with creating scikit estimators

from sklearn.base import BaseEstimator, TransformerMixin

class ChooseFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, chosen_features : list = []):
        self.chosen_features = chosen_features

    def fit(self, X : pd.DataFrame, y : pd.Series = None ):
        # Learns nothing from the data
        return self
    
    def transform(self, X):
        Xc = X.copy()
        return Xc[self.chosen_features]

class ReplaceNA(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_replacement : dict = None):
        self.columns_to_replacement = columns_to_replacement

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        # Learns nothing from the data
        return self
    
    def transform(self, X):
        Xc = X.copy()
        if not self.columns_to_replacement:
            return Xc
        
        missing_col = [col for col in self.columns_to_replacement.keys() if col not in Xc.columns]

        if missing_col:
            raise ValueError(f'Columns not found in input: {missing_col}')


        for column, replacement in self.columns_to_replacement.items():
            

            Xc[column] = Xc[column].fillna(replacement)

        return Xc



class StandardizeColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns = []):
        self.columns = columns

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        Xc = X.copy()
        self.means = Xc[self.columns].mean()
        self.stds = Xc[self.columns].std()
        return self
    
    def transform(self, X : pd.DataFrame):
        Xc = X.copy()

        if not self.columns:
            return Xc
        
        
        missing_col = [col for col in self.columns if col not in Xc.columns]

        if missing_col:
            raise ValueError(f'Columns not found in input: {missing_col}')

    
        for col in self.columns:
            std = self.stds.loc[col]
            if std == 0:
                Xc[col] = 0
            
            else: 
                Xc[col] = (Xc[col] - self.means.loc[col])/std
            

        return Xc












# Cleanup Data

# Select columns we will use in linear regression

def linear_regression_variables(db : pd.DataFrame, columns : list[str]):
    if columns is None or columns == []:
        raise ValueError("No Columns to select")
    
    invalid_columns = [col for col in columns if col not in db.columns]
    if invalid_columns: #python treats empty containers as False or True otherwise
        raise ValueError("One or more inputed columns are not columns in the inputed DataFrame")

    return db[columns].copy()



# Replace na/missing data with a method

from typing import Callable

def replace(db : pd.DataFrame, column_to_method : dict):
    if column_to_method is None :
        raise ValueError("No Columns or Methods specified")
    
    data = db.copy()
    
    for col in column_to_method.keys():
        replacement = column_to_method[col](data[col])
        data[col] = data[col].fillna(replacement)
    return data

# Normalizing function

def normalize_columns(db: pd.DataFrame, columns : list[str]):
    
    if columns is None or columns == []:
        raise ValueError("No columns to normalize")
    
    data = db.copy()
    
    for col in columns:
        col_max = data[col].max()
        col_min = data[col].min()
        if col_min == col_max:
            # want to avoid division by zero
            data[col].nan
        else:
            data[col] = (data[col] - col_min)/(col_max - col_min)
    return data


# Pipeline
from sklearn.pipeline import Pipeline
