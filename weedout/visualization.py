import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def show_removed_outliers(df_original: pd.DataFrame, df_pre_processed: pd.DataFrame):
    """
        This is a visualization function to compare the boxplot spread of the original dataset and 
        the proccessed dataset. 

        @paramters:

            df_original: pd.DataFrame
                The dataframe for the original dataset.

            df_preprocessed: pd.DataFrame
                The dataframe for the preprocessed dataset.

    """

    sns.set(style="whitegrid")

    # Get numerical columns
    numerical_cols = df_original.select_dtypes(include=['int64', 'float64']).columns

    # Define the figure and axes for the subplots
    fig, axes = plt.subplots(nrows=2, ncols=len(numerical_cols), figsize=(50, 50), sharey=True)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot box-plots for original and pre-processed dataframe
    for i, column in enumerate(numerical_cols):
        if i < len(axes) // 2:
            sns.boxplot(data=df_original[column], ax=axes[i], palette="Set2")
            axes[i].set_title(f'Original - {column}')
        else:
            sns.boxplot(data=df_pre_processed[column], ax=axes[i], palette="Set2")
            axes[i].set_title(f'Pre-Processed - {column}')

        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_filtered_correlation_matrix(df: pd.DataFrame):
    """
        This function plots the correlation matrix of the features based onVariance Inflation Factor ( VIF = 1/(1-R^2) ) 
        of each feature column. VIF is a strong indicator of multi-collinearity in our dataframe. 

        Note : The user is expected to remove some of the corelated features based on the VIF value.

        @paramater:

            df : pd.DataFrame
                The dataframe provided by user.
    
    """
    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
    print("Before: ", numerical_columns.shape)
    
    vif = pd.DataFrame()
    vif['Feature'] = numerical_columns.columns
    vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
    
    print("Initial VIF values: ")
    print(vif.sort_values(by="VIF", ascending=False))
    
    infinite_vif_features = vif[vif["VIF"] == np.inf]["Feature"].tolist()
    if infinite_vif_features:
        print(f"\nDropping columns with infinite VIF values: {infinite_vif_features}\n")
        numerical_columns = numerical_columns.drop(columns=infinite_vif_features)
        
    max_vif = 10
    remove_flag = True
    
    while remove_flag:
        vif = pd.DataFrame()
        vif['Feature'] = numerical_columns.columns
        vif["VIF"] = [variance_inflation_factor(numerical_columns.values, i) for i in range(numerical_columns.shape[1])]
        
        max_vif_feature = vif.loc[vif['VIF'].idxmax()]
        
        if max_vif_feature['VIF'] > max_vif:
            numerical_columns = numerical_columns.drop(max_vif_feature['Feature'], axis=1)
            print(f"Removed variable with high VIF {max_vif_feature['Feature']} (VIF={max_vif_feature['VIF']})")
        else:
            remove_flag = False

    print("After: ", numerical_columns.shape)
    
    plt.figure(figsize=(13,10))
    plt.title("VIF")
    sns.heatmap(numerical_columns.corr(),annot=True,fmt='0.2g',cmap='coolwarm',vmin=-1,vmax=1)
    plt.show()