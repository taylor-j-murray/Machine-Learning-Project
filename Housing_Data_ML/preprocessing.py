

import pandas as pd 
import numpy as np
import matplotlib as mp

def PercentageBetween( ser  : pd.Series,a : int, b : int, include_a = True, include_b = True ):
    
    if not pd.api.types.is_numeric_dtype(ser):
        raise TypeError(f"The dtype of the series {ser} is not numeric")
    if a>b:
        raise ValueError(f"{a} is not smaller or equal to {b}")
    
    if not isinstance(a, (int, float)):
        raise TypeError(f"The type of {a} is not float")
    if not isinstance(b, (int, float)):
        raise TypeError(f"The type of {b} is not float")
    
    if include_a is True:
        lower = ser >= a
    elif include_a is False:
        lower= ser > a
    elif type(include_a) is not bool:
        raise TypeError(f"The type of {include_a} is not bool")
    
    if include_b is True:
        upper = ser <= b
    elif include_b is False:
        upper = ser < b
    elif type(include_b) is not bool:
        raise TypeError(f"The type of {include_b} is not bool")
    
    in_between = ser[lower & upper]
    tot_in_between = in_between.count()
    tot_original = ser.count()
    return (tot_in_between/tot_original) * 100





def NormalMetrics( db : pd.DataFrame, n : int):
    columns = db.columns
    metrics = {}
    index_list = [f"plus or minus {i} std's from mean" for i in range(1,n+1)]
    metrics["Standard deviation range"] = index_list
    for col in columns:
        if pd.api.types.is_numeric_dtype(db[col]):
            col_list = []
            assert not col_list
            mean = float(db[col].mean())
            std = float(db[col].std())
            for i in range (1,n+1):
                a = mean - std*i
                b = mean + std*i
                col_list.append(PercentageBetween(db[col], a,b))
            metrics[col] = col_list
        else:
            col_list = []
            assert not col_list
            col_list = [np.nan] * n
            metrics[col] = col_list
            
    normal_metrics = pd.DataFrame(metrics)
    return normal_metrics.set_index("Standard deviation range")

def ZScoreMetrics(db : pd.DataFrame, id_col = None, id_col_idx = False, zero_std_sub = np.nan):
    db = db.copy()
    columns = db.columns
    
    if (id_col is not None) and (id_col not in columns):
        raise ValueError(f"{id_col} is not a feature of the inputted data")
    
    
    numeric_columns = db.select_dtypes(include = np.number).columns
    nonnumeric_columns = columns.difference(numeric_columns)
    
    
    for col in numeric_columns:
        if col == id_col:
            continue
        mean = db[col].mean()
        std = db[col].std()
        if std != 0:
            db['Z-Scores of ' + col] = (db[col] - mean) / std
        else:
            db['Z-Scores of ' + col] = zero_std_sub

    for col in nonnumeric_columns:
        db['Z-Scores of' + col] = np.nan
        
    db = db.rename(columns = lambda col: f"z-scores for {col}")
    
    if not id_col_idx or id_col is None:
        return db
    elif id_col_idx and id_col is not None:
        return db.set_index(id_col)
    
    
def ZScoreFilter(db : pd.DataFrame, bound = 3, col = None, zero_std_sub = np.nan):
    
    if col is not None and col not in db.columns:
        raise ValueError("The inputted column is not a column in the inputted DataFrame")
    db = db.copy()
    numeric_columns = db.select_dtypes(include =np.number).columns
    
    for column in numeric_columns:
        mean = db[column].mean()
        std = db[column].std()
        if std != 0:
            db[column] = db[column].map(lambda x : x if np.abs((x-mean)/std)< bound else 'Fails Z-score bound')
        else:
            db[column] = zero_std_sub
    
    if col is None:
        return db
    else:
        return db[col]
    

        


    
    
    
    
    
#def ZScoreFilter(db: pd.DataFrame, col: str, lower_bound = -3, upper_bound =3, invert = False):
    


#df = pd.DataFrame({
    #'A': np.random.normal(0, 1, 1000),
    #'B': np.random.normal(5, 2, 1000),
    #'C': ['cat', 'dog', 'mouse'] * 333 + ['cat']
#})
#print(NormalMetrics(df, 3))

def IQRBounds(ser : pd.Series):
    Q1 = ser.quantile(.25)
    Q3 = ser.quantile(.75)
    IQR = Q3-Q1
    return {'IQR': IQR,'Lower Bound' : Q1-IQR*1.5, 'Upper Bound': Q3 + IQR*1.5}


def IQRMetrics(db : pd.DataFrame):
    db = db.copy()
    numeric_columns = db.select_dtypes(include = np.number).columns
    metrics ={}
    if numeric_columns.empty:
        raise ValueError("The DataFrame inputted does not have any numeric columns")
    for col in numeric_columns:
        iqrbounds = IQRBounds(db[col])
        col_list = [iqrbounds['IQR'],
                    iqrbounds['Lower Bound'],
                    iqrbounds['Upper Bound']
                    ]
        metrics['IQR- Metrics for '+ col] = col_list
    
    idx = ['IQR', 'Lower Bound', 'Upper Bound']
    return pd.DataFrame(metrics, index = idx)
        

def IQRFlag(db : pd.DataFrame, disclude : list[str] = [], invert = False, filter = False):
    db = db.copy()
    numeric_columns = [ col for col in db.select_dtypes(include = np.number).columns if col not in disclude]
    for col in numeric_columns:
        bounds = IQRBounds(db[col])
        lower_bound = bounds['Lower Bound']
        upper_bound = bounds['Upper Bound']
        mask = (db[col] > upper_bound) | (db[col] < lower_bound)
        if invert: 
            mask = ~mask
        elif not invert: 
            pass
        
        if filter == False:
            db['Flag_IQR_for_' + col] = mask
            
        elif filter == True:
            db = db[mask]

    return db
    

    

    





# Would be interesting to add a function that determines which outlier method to use based on columns data structure.
