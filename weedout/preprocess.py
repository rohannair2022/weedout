import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder, FunctionTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from typing import List, Optional, Tuple
from sklearn.utils import resample



def check_target_variable(df: pd.DataFrame, target_name: str) -> bool:
    """
        The function returns true if the given target name exists in the dataframe. It returns false
        if the target name is not in the dataframe. 

        @paramters: 
            df: pd.DataFrame
                Given dataframe.
            
            target_name: str
                Given target Name 
        @return:

            bool:
                Indicates if target name exists of not.
    """
    seen = False 
    for col in df.columns:
        if col == target_name:
            seen = True 
            break 
    return seen


def check_duplicate_columns(file_path: str) -> List[str]:
    """
        The function checks if the dataset has multiple same column names.

        @paramter:

            file_path:
                The path to the csv file 

        @return:

            List[str]:
                A list of all the duplicate column names.

    """
    if file_path.lower().endswith('.csv'):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            seen = set()
            duplicates = set()
            
            for header in headers:
                if header in seen:
                    duplicates.add(header)
                seen.add(header)
            
            if duplicates:
                return list(duplicates)
            else:
                return []
    else:
        raise Exception('Wrong File Type. Only CSV allowed.')


def initial_check_dt(file_path: str, target_variable: str, columns_to_drop: List[str]) -> Optional[pd.DataFrame]:
    """
        The function initially checks if the file provided is a valid CSV file that can be parsed through. It checks 
        and drops duplicate columns, verifies the target variable, and drops columns with exceedingly high
        null values and untouched columns.

        @parameters:

            file_path : str
                The path to the csv file.

            target_variable : str  
                The target column of your dataset.

            untouched_columns : List[str] 
                The list of columns that are not going to be modified. These 
                would generally be columns that have to do with IDw.
        
        @return:

            Optional[pd.DataFrame] 
                It will return the modified dataframe if the function is successful.

    """
    if file_path.lower().endswith('.csv'):

        try:
            duplicate = check_duplicate_columns(file_path)
            if duplicate:
                raise Exception("Duplicate columns exist in the file:", duplicate)
            df = pd.read_csv(file_path)
        except pd.errors.ParserError:
            raise Exception('File cannot be parsed as a CSV.') from None
        except Exception as e:
            raise Exception(f"An error occurred: {e}") from e
        
        id_drop = [col for col in df.columns if col in columns_to_drop]
        if id_drop:
            df = df.drop(columns=id_drop)
            print(f"Dropped untouched columns: {', '.join(id_drop)}")
            
        drop_columns = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            total_values = len(df[col])
            missing_percentage = (missing_count / total_values) * 100

            if missing_percentage > 40:
                drop_columns.append(col)

        print("Dropped columns with high missing data: ", list(drop_columns))
        df = df.drop(columns=drop_columns)
        
        df_columns = df.columns.tolist() 
        keep_indices = []

        for index, col in enumerate(df.columns):
            keep_indices.append(index)

        df = df.iloc[:, keep_indices]
        
        if target_variable in df_columns:
            print(f"The target variable '{target_variable}' is present in the DataFrame.")
        else:
            raise Exception (f"The target variable '{target_variable}' is missing in the DataFrame.")
        
        print("Initial checks and changes made")
        return df
    
    else:
        raise Exception("Oops :( The file is not a CSV file!")

def cross_sectional_imputation(cross_sectional_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
        The function works on imputation for the given cross sectional dataframe. For columns with
        numeric values, the null values of that column are imputed/filled with the mean (average) value 
        of the column. For columns with string values, the null values of that column are imputed/filled
        with the mode (most frequent) value of that column. 

        @parameters:

            cross_sectional_df : pd.DataFrame:
                The dataframe of the cross-sectional dataset with the target column included.

            target_name: str:
                The name of the target column
        
        @return:
        
            Optional[pd.DataFrame] 
                It will return the imputed dataframe if the function is successful.

    """
    df = cross_sectional_df.copy()
    
    print("Total Null value counts before imputation: \n",df.isnull().sum())

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != target_name:
            df[column] = df[column].fillna(df[column].mean())
            df[column] = df[column].astype(float)
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    
    print("\nTotal Null value counts after imputation: \n",df.isnull().sum())
    return df

    
def time_series_imputation(time_series_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
        The function works on imputation for the given time series dataframe. For columns with
        numeric values, the null values of that column are imputed/filled through the method of linear 
        interpolation. For columns with string values, the null values of that column are imputed/filled
        with the mode (most frequent) value of that column. 

        Note: If the first row contains a null value in the numerical column then the imputation will not 
        work. 

        @parameters:

            time_series_df : pd.DataFrame:
                The dataframe of the time series dataset with the target column included.
            
            target_name: str:
                The name of the target column
        
        @return:
        
            Optional[pd.DataFrame] 
                It will return the imputed dataframe if the function is successful.
    """
    df = time_series_df.copy()

    first_row = df.iloc[0]
    has_nan_and_is_numeric = first_row.isnull() & first_row.apply(lambda x: isinstance(x, (float, int)))
    if has_nan_and_is_numeric.any():
        raise Exception('The first row cannot have a numeric null value.')
    
    print("Total Null value counts before imputation: \n",df.isnull().sum())

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != target_name:
            df[column] = df[column].interpolate(method='linear')
            df[column] = df[column].astype(float)
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
        
    print("\nTotal Null value counts after imputation: \n",df.isnull().sum())
    return df


def handle_imbalanced_data(df: pd.DataFrame, target_variable: str, strategy = "smote", k_neighbors=2, untouched_columns: List[str]=[]) -> pd.DataFrame:
    """
        The function balances a dataframe defined through a given sampling strategy to be
        ran on a classfication model.

        Note: Choosing "smote" as your strategy would give you an oversampled encoded dataframe. 

        @paramters:

            df: pd.DataFrame
                The dataframe for the given dataset.
            
            target_variable: str
                The name of the target column in the dataframe

            stratergy: str: default = SMOTE
                The name of the sampling stratergy:
                    - oversampling 
                    - undersampling 
                    - smote 
            
            k_neighbours: int: default = 2
                The number of neighbours for the Smote Stratergy

            untouched_columns : List[str]
                The list of column names that should not be encoded.
        
        @return:

            pd.DataFrame
                The balanced dataframe after following the sampling stratergy.
    
    """

    if not check_target_variable(df, target_variable):
        raise Exception(f"The target variable '{target_variable}' is missing in the DataFrame.")

    if strategy == 'oversampling':
        sampler = RandomOverSampler()
    elif strategy == 'undersampling':
        sampler = RandomUnderSampler()
    elif strategy == 'smote':
        sampler = SMOTE(k_neighbors=k_neighbors)
    else:
        raise ValueError("Invalid strategy. Choose from 'oversampling', 'undersampling', or 'smote'.")
    
    print(f'The distribution of the target column prior to sampling: {df[target_variable].value_counts}')
    
    if strategy == "smote":

        features, target = separate_target_column(df, target_variable)

        features = encoding(features, untouched_columns)

        columns_to_drop = [col for col in untouched_columns if col in features.columns]
        features_new = features.drop(columns=columns_to_drop)

        post_df = combine(features_new,target)

        X_res, y_res= separate_target_column(post_df,target_variable)

        X_res, y_res = sampler.fit_resample(X_res, y_res)

        df_balanced = pd.concat([X_res, y_res], axis=1)

        df_balanced = pd.concat([df_balanced, df[untouched_columns].reset_index(drop=True)], axis=1)

        for column in df_balanced.columns:
            if column in untouched_columns:
                df_balanced[column] = df_balanced[column].fillna(df_balanced[column].mode()[0])

    else:

        X_res, y_res= separate_target_column(df,target_variable)

        X_res, y_res = sampler.fit_resample(X_res, y_res)

        df_balanced = pd.concat([X_res, y_res], axis=1)

    print(f'The distribution of the target column after sampling: {df_balanced[target_variable].value_counts}')
    
    return df_balanced

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
        The function detects outliers outside the range of 2.575 (99% CI) and removes the corresponding
        observations/rows if the outlier is present in more than 10% of the column. 

        @paramters:

            df: pd.DataFrame
                The dataframe for the given dataset.
        
        @return:

            pd.DataFrame
                The balanced dataframe after following the sampling stratergy.

    """
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    df_original = df.copy()

    outlier_counts = pd.Series(0, index=df_original.index)
    threshold = 2.575 # Indicating a 99% CI
    col_count = len(numerical_features)

    for col in numerical_features:
        z_scores = np.abs(stats.zscore(df_original[col]))
        outliers = z_scores > threshold
        outlier_counts += outliers

    # Remove rows with a significant number of outliers.
    df_pre_processed = df_original[outlier_counts < col_count*0.1]

    print(f"\nOriginal data count: {len(df_original)}")
    print(f"After outlier removal data count: {len(df_pre_processed)}")

    return df_pre_processed

def separate_target_column(df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

        This function separates the given target column from the data frame.    

        @paramters:

            df: pd.DataFrame
                The dataframe for the given dataset that contains the target column as well.
            
            target_variable: str
                The name of the target column in the dataframe.
        
        @return:

            Tuple(pd.DataFrame, pd.DataFrame):
                Tuple[0] : The dataframe with all columns except the target column
                Tuple[1] : The target column dataframe.


    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError
        
        if not check_target_variable(df, target_variable):
            raise ValueError

        target = df[[target_variable]]
        remaining_df = df.drop(columns=[target_variable])
        return remaining_df, target
    
    except TypeError:
        raise Exception('Data type error.Please provide the right type of dataframe.')
    
    except ValueError:
        raise Exception('Target Column does not Exist. Please provide the right one.')


def encoding(features: pd.DataFrame, untouched_columns: List[str]=[]) -> pd.DataFrame:
    """
        The function encodes the object type columns in the given data frame. If the number of
        attributes in a column exceeds more than 3, then the function performs Label Encoding. If it does not, 
        then it performs One Hot Encoding. 

        @parameters:
            features : str
                The data frame consisting of all the features (excluding the target column).
            
            untouched_columns : str
                The list consists of all the 
        
        @return:
            pd.DataFrame:
                The dataframe consisting of all the encoded features. 
    """
    print("Before encoding:", features.columns)
    for column in features.columns:
        if features[column].dtype == "object" and column not in untouched_columns:
            if features[column].nunique() > 3:
                encoder = LabelEncoder()
                features[column] = encoder.fit_transform(features[column])
            else:
                encoder = OneHotEncoder(sparse_output=False)
                encoded = encoder.fit_transform(features[[column]])
                encoded_features = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=features.index)
                features.drop(columns=[column], inplace=True)
                features[encoded_features.columns] = encoded_features
    
    print("After encoding:", features.columns)

    return features


def feature_scaling(features: pd.DataFrame, unscale_columns: List[str]) -> pd.DataFrame:
    """
        The function scales continuous-numerical values of the dataset. If the values in the column have a normal
        distribution then we do StandardScaling. If it does not, then we do MinMaxScaling
        Note : If the unique value count of the column is greater than 3, only then will the function apply 
        scaling to that column.

        @parameters:

            features:
                The dataframe containing the features to be scaled.
            
            unscale_columns:
                The list of column names that represent the columns that should not be scaled.


        @return:

            pd.DataFrame:   
                The dataframe containing the scaled features.

    
    """
    df = features.copy()
    
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

    def is_normal(column, alpha=0.05, max_sample_size=5000):
        column = column.dropna()

        if len(column) > max_sample_size:
            column = resample(column, n_samples=max_sample_size, random_state=42)

        anderson_result = stats.anderson(column, dist='norm')
        anderson_normal = anderson_result.statistic < anderson_result.critical_values[2]  

        skewness = stats.skew(column)
        kurtosis = stats.kurtosis(column)

        skew_normal = abs(skewness) < 0.5
        kurtosis_normal = abs(kurtosis) < 0.5

        # Return True if all conditions are met
        return anderson_normal and skew_normal and kurtosis_normal

    
    Minmaxscaler_algorithms = []
    Standardscaler_algorithms = []

    
    for col in numerical_columns:
        if col not in unscale_columns:

            unique_count = df[col].nunique()

            if is_normal(df[col]):
                
                if unique_count > 3: 
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    Standardscaler_algorithms.append(col)

            else:

                if unique_count > 3: 
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    Minmaxscaler_algorithms.append(col)
            
                        
    print("\nMinmax scaled columns: \n", Minmaxscaler_algorithms)
    print("\nStandardized columns: \n", Standardscaler_algorithms)
    
    df_scaled = pd.concat([df[numerical_columns], df[non_numerical_columns]], axis=1)
    
    df_scaled = df_scaled.reset_index(drop=True)

    return df_scaled

def combine (features: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """
        The functions combines the features dataframe and target dataframe into one dataframe.

        @paramters:
            features: pd.DataFrame
                The dataframe that consists of all the features
        
            target: pd.DataFrame
                The dataframe that consists of all the targets.
        
        @return:
            
            pd.DataFrame:
                The combined dataframe that consists of both the feature dataframe and the target dataframe.
    
    """
    try:
        if not isinstance(features, pd.DataFrame) and not isinstance(target, pd.DataFrame):
            raise TypeError
        combined_df = pd.concat([features, target], axis=1)
        return combined_df
    
    except TypeError:
        raise Exception('Data type error.Please provide the right type of dataframe.')


def preprocess_pipeline(file_path: str, target_column: str, dropped_columns: List[str]=[], untouched_columns: List[str]=[], type_dataset: int=0, sampling: int=0, classfication: int=1, strategy_sample="smote"):
    """
        This function is a stand-alone pipeline used for the free website template where you can 
        interact with a GUI.
        
        @parameters:

            file_path: str 
                The path to the dataset.

            target_column: str 
                The name of the target variable.

            dropped_columns: List[str]
                The list of column names the client wants dropped from the dataset.

            untouched_columns: List[str]
                The list of columns that should be not scaled/encoded.

            type_dataset: int
                The type of dataset provided:
                    - 0 -> Cross Sectional
                    - 1 -> Time Series
            
            sampling: int 
                Indicates whether the client wants to perform sampling or not
                    - 0 -> No sampling
                    - 1 -> Sampling 
                    
            classification: int 
                The type of model that the dataset will be trained on 
                Note: The pipeline does not perform sampling for regression models.
                    - 0 -> regression
                    - 1 -> classification

            stratergy_sample: str: default = SMOTE
                The name of the sampling stratergy:
                    - oversampling 
                    - undersampling 
                    - smote 

        @return:

            pd.DataFrame:
                The proccessed dataframe.

    """
    print("\nInitial Check")
    df = initial_check_dt(file_path, target_column, dropped_columns)
    print("\n-----------------------------Initial check done-----------------------------------------------\n")
    
    print("\nImputation")
    if not type_dataset:
        df = cross_sectional_imputation(df, target_column)
        print("\n-----------------------------Cross Sectional Imputation Done-----------------------------------------------\n")
    elif type_dataset == 1:
        df = time_series_imputation(df, target_column)
        print("\n-----------------------------Time Series Imputation Done-----------------------------------------------\n")

    if sampling:
        if classfication:
            if df[target_column].nunique() > 2:
                print("\n-----------------------------Balancing Operations not suported for Multi-class Classification-----------------------------------------------\n")
            else:
                if strategy_sample == "smote":            
                    print("\nHandling Imbalanced Data")
                    df = handle_imbalanced_data(df, target_column, strategy_sample, untouched_columns=untouched_columns)
                    print("\n-----------------------------Balanced and Encoded the data-----------------------------------------------\n")
                else:
                    print("\nHandling Imbalanced Data")
                    df = handle_imbalanced_data(df, target_column, strategy_sample)
                    print("\n-----------------------------Balanced the data-----------------------------------------------\n")
        else:
           print("\n-----------------------------Balancing Operations not suported for Regression Models-----------------------------------------------")
           print("\n-----------------------------Did not Balance the data-----------------------------------------------\n")

    print("\nRemoving Outliers")
    df = remove_outliers(df)
    print("\n-----------------------------Removed Outliers-----------------------------------------------\n")
    
    print("\nSeparating Input and Output")
    remaining_df, target = separate_target_column(df, target_column)
    print("\n-----------------------------Separated the input and output-----------------------------------------------\n")
    
    
    if strategy_sample != "smote" or df[target_column].nunique() > 2:
        print("\nEncoding Data")
        remaining_df = encoding(remaining_df, untouched_columns)
        print("\n-----------------------------Encoded the data-----------------------------------------------\n")
    
    print("\nFeature Scaling")
    preprocessed_df = feature_scaling(remaining_df, untouched_columns)
    print("\n-----------------------------Feature Scaling Done-----------------------------------------------\n")
    
    print("\nCombining Feature and Target")
    combined_df = combine(preprocessed_df,target)
    print("\n-----------------------------Combining Features Done-----------------------------------------------\n")
    
    return combined_df
