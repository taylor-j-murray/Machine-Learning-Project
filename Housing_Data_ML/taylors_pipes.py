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

    def __init__(self, columns = None):
        self.columns = columns

    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        Xc = X.copy()
        self.means = Xc[self.columns].mean()
        self.stds = Xc[self.columns].std()
        return self
    
    def transform(self, X : pd.DataFrame):
        Xc = X.copy()

        if self.columns is None :
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
        

class LogTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns = None, replace = True, base : int = None):
        self.columns = columns
        self.replace = replace
        self.base = base # This can be a non-positive real number, but we restrict to int for now
        
        
        if not isinstance(self.replace, bool):
            raise TypeError(f'The replace parameter must be a boolean, but has received {replace}')
        if base is not None and base <= 0:
            raise ValueError(f'The base cannot be non-positive, but has received {base}')
        
    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        # No learning is done
        return self
    
    def transform(self, X : pd.DataFrame):
        X = X.copy()
        
        if self.columns is None:
            return X
        
        missing_col = [col for col in self.columns if col not in X.columns]
        
        if missing_col:
            raise ValueError(f'Columns not found in input: {missing_col}')
        
        if self.base is None:
            D = 1
        else : 
            D = np.log(self.base)
            
        for col in self.columns:
            
            if (X[col]<=0).any():
                raise ValueError(f'The column {col} contains non-positive values- log is undefined.')
            
            log_vals = np.log(X[col])/D
            
            
            if self.replace:
                X[col] = log_vals
            else: 
                X[f'log_{self.base if self.base is not None else np.e}({col})'] = log_vals
                # Maybe add optional parameter allowing a chosen name for 
                # the transformed columns

        return X

class ArithmeticTransformer(BaseEstimator, TransformerMixin):
    ALLOWED_OPS ={'plus', 'minus', 'multiply', 'divide'}
    
    def __init__(self, op : str, new_column_name: str, columns = None ):
        # Right now simple arithmetic is allowed, later might add
        # options for selecting exponents for mult/div and coefficients
        # for add/sub
        # for now subtraction and division will only be able to take in
        # two columns at a time since in these cases order matter.
        
        
        
        self.columns = columns
        self.op = op
        self.new_column_name = new_column_name
        
        if op not in self.ALLOWED_OPS:
            raise ValueError (f"{op} must be a value from {self.ALLOWED_OPS}")
        
        if self.columns is None: 
            pass
        elif not isinstance(self.columns, list):
            raise TypeError(f'{self.columns} is not None nor a list')
        elif (op == 'minus' or op == 'divide') and len(self.columns) < 2:
            raise ValueError (f'''The operator {op} only allows for two columns at a time 
                                instead of {len(self.columns)} columns''')
            
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # No learning done
        return self
    
    def transform(self, X : pd.DataFrame):
        
        X = X.copy()
        
        if self.columns is None:
            # No columns specified returns X unchanged
            return X
        
        missing_col = [col for col in self.columns if col not in X.columns]
        
        if missing_col:
            raise ValueError(f'Columns not found in input: {missing_col}')
        
        if len(self.columns) < 2:
            return X
        
        
        #Addition
        if self.op == 'plus':
            X[self.new_column_name] = 0 # 0 is the additive identity
    
            for col in self.columns:
                X[self.new_column_name] = X[self.new_column_name] + X[col]
                
                
        # Subtraction 
        # Here the convention is that the first element of self.columns will be subtracted by
        # the second element of self.columns
        
        elif self.op == 'minus':
            X[self.new_column_name] = X[self.columns[0]] - X[self.columns[1]] 
            # only two elements in self.columns for 'minus', so this is fine
            
        # Multiplication
        elif self.op == 'multiply':
            X[self.new_column_name] = 1 # 1 instead of 0 since 1 is the multiplicative identity
    
            for col in self.columns:
                X[self.new_column_name] = X[self.new_column_name] * X[col]
                
        # Division
        
        elif self.op == 'divide':
            if (X[self.columns[1]] == 0).any():
                raise ValueError(f'The column {self.columns[1]} has a value of 0 - division is not well defined.')
            
            X[self.new_column_name] = X[self.columns[0]] / X[self.columns[1]]
            
        else:
            raise ValueError(f"Invalid operator: {self.op} ")
        
        return X
            
                
class Scale(BaseEstimator,TransformerMixin):
    
    def __init__(self, col : str, scale_by : int = 1 ):
        # Thinking about a replace option, but in general
        # we don't usually create a new column with a scaled
        # column; we usually just replace the original column 
        # with the scaled version.
        self.scale_by = scale_by
        self.col = col
    
    def fit(self, X : pd.DataFrame, y : pd.Series):
        # No learning done.
        return self
    
    def transform(self, X : pd.DataFrame):
        if self.col not in X.columns:
            raise ValueError(f'The input {self.col} is not a valid column in the inputted DataFrame')
        
        X = X.copy()
        X[self.col] = self.scale_by * X[self.col]
        return X
    
class OneHotEncode(BaseEstimator,TransformerMixin):
    
    def __init__(self, col : str, drop : bool):
        self.col = col
        self.drop = drop
        
    def fit(self, X : pd.DataFrame, y: pd.Series):
        if self.col not in X.columns:
            raise ValueError(f'The input {self.col} is not a valid column in the inputted DataFrame')
        
        self.categories = sorted(X[self.col].dropna().unique())
        return self
    
    def transform(self, X :pd.DataFrame):
        if self.col not in X.columns:
            raise ValueError(f'The input {self.col} is not a valid column in the inputted DataFrame')
        
        X = X.copy()
        
        for val in self.categories:
            col_name = f'{self.col}_{val}'
            mask = (X[self.col] == val)
            if col_name in X.columns:
                raise NameError( f'''{col_name} is a column in the inputted DataFrame;
                                rename {col_name} to another value.''')
            else:
                X[col_name] = mask.astype(int)
        
        
        if self.drop:
            X = X.drop(columns = [self.col])
        
        return X