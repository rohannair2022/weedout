import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def filtered_correlation_matrix(df: pd.DataFrame):
    """
        This function prints the Variance Inflation Factor ( VIF = 1/(1-R^2) ) of each feature column. VIF 
        is a strong indicator of multi-collinearity in our data frame. 

        Note : The user is expected to remove some of the correlated features based on the VIF value.

        @parameter:

            df : pd.DataFrame
                The dataframe provided by user.
        
    """

    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    print("Before: ", numerical_columns.shape)
    
    vif = pd.DataFrame()
    vif['Feature'] = numerical_columns.columns
    vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
    
    print("Initial VIF values: ")
    print(vif.sort_values(by="VIF", ascending=False))


def display(file_path: str ,df: pd.DataFrame) -> None:
    """
        The function prints the info of the original dataset in the given file path in comparison to
        the preprocessed dataframe.

        @paramters:
            file_path: str 

                The path to the original file.

            df: pd.DataFrame 

                The preproccsed dataframe.

    """
    original_data=pd.read_csv(file_path)
    processed_data=df
    print("\nOriginal Data\n")
    original_data.info()
    print("\nProcessed Data\n")
    processed_data.info()