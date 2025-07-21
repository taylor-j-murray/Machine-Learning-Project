import pandas as pd
import numpy as np
import matplotlib as mp
import preprocessing
import utilities


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocessing import  linear_regression_variables, normalize_columns 
from sklearn.pipeline import Pipeline

# Read in Data 
db = pd.read_csv('/Users/tayma/Downloads/housing_data_kaggle.csv')
# Create a copy of data
data = db.copy()

# Split the data into a training set and a test set

split_data = utilities.split(data, id_column = 'id', test_size = 0.2)
test_set, train_set = split_data

# Create a label pd.Series 
y = train_set['price']

# Drop label column from training set
X = train_set.drop('price', axis =1)

# Create list of estimators for scikit Pipeline

model = LinearRegression()
chosen_features = ['sqfeet', 'beds', 'baths']
col_na_replacement = {}
for col in chosen_features:
    col_na_replacement[col] = X[col]
columns_to_standardize = ['sqfeet']

estimators = [('choose features', preprocessing.ChooseFeatures(chosen_features = chosen_features)), 
              ('replace_na',preprocessing.ReplaceNA(columns_to_replacement = col_na_replacement)), 
              ('standardize', preprocessing.StandardizeColumns(columns = columns_to_standardize)), 
              ('chosen_model', model)]

# Feed estimators into Pipeline

pipe = Pipeline(estimators)
fitted = pipe.fit(X,y)
predicts = fitted.predict(X)
print(predicts)



