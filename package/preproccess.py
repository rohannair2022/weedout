import numpy as np 
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from statistics import mode


# Initial Read for the CSV file from User
def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# def initial_check(file_path: str) -> pd.DataFrame:
    # Check if the dataset is missing more than 40% of its values. (Nan)
    # CSV File (Correct DataType)
    # Multiple same Feature names 
    # Missing Target variable name given by User


# Split the Dataset into the desired Features and Target.
def split_data(target_name: str):
    # Setting up the data frame
    data_frame = read_csv('target_name')

    # Feature Variables
    features = data_frame(target_name, axis=1)  

    # Target Variable 
    target = data_frame[target_name]  

    return [features, target]


# Imputing -> type 
# Random Input (0) : Numerical: Mean, Categorical: Mode
# Time Series Data Random(1): Numerical: Mean, Categorical: Mode
# Time Series Data in Order (2): Numerical: Linear Interpolation, Categorical: Mode
# @ Linear Interpolation: Takes the average of the previous and next element. Implemented with inerpld.
def imputing(type: int, features: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    if(type == 0 or type == 1):
        for column in target.columns:
            if target[column].dtype != "object":
                mean_value = target[column].mean().astype('float32')
                target[column] = target[column].fillna(mean_value)
            else:
                target[column] = target[column].fillna(mode(target[column]))
    else:   
        for column in target.columns:
            if target[column].dtype != "object":
                for index, value in target[column].items():
                    if pd.isna(value):
                        imputed_value = linear_interpolated_val()
                        target.at[index, column] = imputed_value
            else:
                target[column] = target[column].fillna(mode(target[column]))

def linear_interpolated_val()
