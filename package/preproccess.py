import numpy as np 
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
def split_data(file_path: str, target_name: str):
    # Setting up the data frame
    data_frame = read_csv('file_path')

    # Feature Variables
    features = data_frame.drop(target_name, axis=1)  

    # Target Variable 
    target = data_frame[target_name]  

    return [features, target]

# Encoding:
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


# Imputing (Filling Missing values) -> type 
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

# Helper Function: Required for Imputation.
# Recurssion for finding linear interpolated value:
def linear_interpolated_val(target: pd.DataFrame, column: str, index: int):
    """
        Recursively find the linear interpolated value for a given index in a DataFrame column.

        @target (pd.DataFrame): The input DataFrame.
        @column (str): The name of the column to interpolate.
        @index (int): The index of the value to interpolate.

        @return(float): The interpolated value.

        >>> df = pd.DataFrame({'A': [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0]})
        >>> linear_interpolated_val(df, 'A', 0)
        1.0
        >>> linear_interpolated_val(df, 'A', 1)
        2.5
        >>> linear_interpolated_val(df, 'A', 2)
        2.5
        >>> linear_interpolated_val(df, 'A', 6)
        7.0
    """
    def recurse_search(index: int, type: str, max: int):
        if type == "negative":
            if(index == 0):
                return 0
            if pd.isna(target.iloc[index-1][column]):
                return recurse_search(index-1, "negative", max)
            else:
                return target.iloc[index-1][column]
        else:
            if(index == max - 1):
                return 0
            if pd.isna(target.iloc[index+1][column]):
                 return recurse_search(index+1, "positive", max)
            else:
                return target.iloc[index+1][column]

    max = len(target)
    if index == 0:
        if not pd.isna(target.iloc[index][column]):
            return target.iloc[index][column]
        else:
            return recurse_search(index, "positive", max)
    elif index == max - 1:
        if not pd.isna(target.iloc[index][column]):
            return target.iloc[index][column]
        else:
            return recurse_search(index, "negative", max)
    else:
        return (recurse_search(index, "positive", max) + recurse_search(index, "negative", max))/2


if __name__ == "__main__":
    import doctest
    doctest.testmod()