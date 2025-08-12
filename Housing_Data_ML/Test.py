

import pandas as pd
import numpy as np
import matplotlib as mp
import preprocessing


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

X = pd.DataFrame({'A': [1,5,3], 'B': [1,np.nan,1]})

chosen_features = ['A', 'B']
column_to_replacement = {'A' : 0, 'B': 0}
estimators = [ ('choose_columns', preprocessing.ChooseFeatures(chosen_features)),('replace_na', preprocessing.ReplaceNA(column_to_replacement))]
pipe = Pipeline(estimators)

new_X = pipe.fit_transform(X)
print(new_X)
print(X)

X