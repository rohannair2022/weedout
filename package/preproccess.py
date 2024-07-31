import numpy as np 
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from statistics import mode
from typing import Literal


# Initial Read for the CSV file from User
def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# def initial_check(file_path: str) -> pd.DataFrame:
    # Check if the dataset is missing more than 40% of its values. (Nan)
    # CSV File (Correct DataType)
    # Multiple same Feature names 
    # Missing Target variable name given by User


# Type of Classfication: 0 - Binary | 1 - Multi
def type_classification(type: int, df: pd.DataFrame, target: str) -> None:
    return 0 if len(df[target].unique()) == 2 else 1

# Balancing dataset:
# We have to keep in mind that this would only be for Classfication and not Regression
# Therefore we have two types of datasets to tackle: Binary classification and Multi-classification. 
"""def balancing(type: int, df: pd.DataFrame) -> None:
    if(not type):"""


# Split the Dataset into the desired Features and Target.
def split_data(file_path: str, target_name: str):
    # Setting up the data frame
    data_frame = read_csv('file_path')

    # Feature Variables
    features = data_frame.drop(target_name, axis=1)  

    # Target Variable 
    target = data_frame[target_name]  

    return [features, target]

# Encoding -> Inplace:
# If the number of unique components in the object datatype column exceeds 3 then we do Label Encoding.
def encoding(features: pd.DataFrame) -> None:
    for column in features.columns:
        if features[column].dtype == "object":
            if features[column].nunique() > 3:
                encoder = LabelEncoder()
                features[column] = encoder.fit_transform(features[column])
            else:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(features[[column]])
                encoded_features = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=features.index)
                features.drop(columns=[column], inplace=True)
                features[encoded_features.columns] = encoded_features



# Imputing (Filling Missing values) -> Inplace:
# Random Input (0) : Numerical: Mean, Categorical: Mode
# Time Series Data in Order (1): Numerical: Linear Interpolation, Categorical: Mode
# @ Linear Interpolation: Takes the average of the previous and next element. Implemented with inerpld.
def imputing(type_impute: int, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    if type_impute == 0:
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
    elif type_impute == 1:
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].interpolate(method='linear')
            else:
                df[column] = df[column].fillna(df[column].mode()[0])
    else:
        raise ValueError("Invalid type. Expected 0, 1, or 2.")
    return df

if __name__ == "__main__":
    import doctest
    doctest.testmod()