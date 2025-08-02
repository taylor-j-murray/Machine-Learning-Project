import pandas as pd

# In scikit-learn .fit() extracts info from data that is to be implemented in a Transformers .transform() or a Predictors .predict().
#.fit() should learn something from the data, learn a parameter like mean or linear regression coefficients.

class ColumnTransformer:
    def __init__(self, col_to_method : dict = None, col_to_learn : dict = None):
        self.col_to_method = col_to_method
        self.col_to_learn = col_to_learn
        self.learned = []



    def fit(self, X : pd.DataFrame, y : pd.Series = None):
        # Does nothing if there is nothing to learn
        if self.col_to_learn is None:
            return self
        Xc = X.copy()
        for col, func in self.col_to_learn.items():
            self.learned.append(func(Xc[col]))
        return self
            
            

    def transform(self, db : pd.DataFrame):
        
        data = db.copy()
        for col, func in self.col_to_method.items():
            if col in data.columns:
                data[col] = func(data[col])
            else:
                raise ValueError(f"Not all columns in col_to_method are a column in the provided DataFrame")
        return data

        
#Add code to allow for tranformations that need fit. What are these types of methods usually called?
        
