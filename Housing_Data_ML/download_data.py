import kaggle as kg
import os
import pandas as pd
import numpy as np
import matplotlib as mp

os.environ['KAGGLE_USERNAME'] ='taylorjmurray' #os.environ is a mapping object, think dictionary. os.environ is created as soon as os module is imported.
os.environ['KAGGLE_KEY'] = '9514abe911ad574d3d1f13f9dfac480e'
kg.api.authenticate()
#download_path = "/Users/tayma/datasets" # Windows 
download_path = "/Users/taylormurray/datasets" #MaciOS
kg.api.dataset_download_files(dataset= 'austinreese/usa-housing-listings', path = download_path, unzip = True)
file_path = os.path.join(download_path, 'housing.csv') #joins the download_path and 'housing.csv' and returns it as a new path
