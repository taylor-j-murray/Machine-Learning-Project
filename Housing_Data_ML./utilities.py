
import pandas as pd
import hashlib 
import numpy as np
import matplotlib as mp
 


def hash_to_value(value):
    return int(hashlib.md5(str(value).encode()).hexdigest(),16)%(10**8)/10**8

# We describe hash_to_value
# str(value).encode() transforms the string to a UTF-8 byte string which is required for hashing
# hashlib.md5(...).hexdigest() converts the UTF-8 byte string to a hexstring
# int(..., 16) coverts the hex string to a base 10 integer
# % 10**8 turns int(...,16) into a 8 digit integer
# / 10**8 gives us a number in [0,1) which is ideal for fair splitting and becaus
# the key point is that the result is a determinisitic float, meaning it will not change output for any run.
# More over MD5 distributes inputs uniformly and randomly over its output range. This allows us to use test_size = 0.2
# in the function split below with certainty that the test set will have size roughly 20% of the original.

def split(df, id_column, test_size = 0.2):
    df = df.copy() # Avoids modifying the dataframe
    df['hash_value'] =  df[id_column].apply(hash_to_value) # Creates a new column called 'hash_value' that applies hash_to_value to every element in the id_column
    test_mask = df['hash_value'] < test_size # Creates a mask (i.e a boolean array) picking out the instances whose hash_value is less than test_size
    test_set = df[test_mask] # returns a subset of the dataset whose instances have hash_value < test_size
    train_set = df[~test_mask] # returns a subset of the dataset whose instances have hash_value >= test_size
    return test_set, train_set


