import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Pandas Functions
# DataFrame Creation
def create_dataframe(data, columns=None):
    return pd.DataFrame(data, columns=columns)


# Data Inspection
def get_head(df, n=5):
    return df.head(n)

def get_tail(df, n=5):
    return df.tail(n)

def get_info(df):
    return df.info()

def describe_statistics(df):
    return df.describe()

def get_shape(df):
    return df.shape

def get_dimensions(df):
    return df.ndim

def get_column_labels(df):
    return df.columns

def get_index_labels(df):
    return df.index


# Data Selection
def select_column(df, column_name):
    return df[column_name]

def label_based_selection(df, label):
    return df.loc[label]

def integer_based_selection(df, position):
    return df.iloc[position]


# Data Filtering
def filter_data(df, condition):
    return df[condition]


# Data Manipulation
def drop_column(df, column_name):
    return df.drop(columns=[column_name])

def rename_column(df, old_name, new_name):
    return df.rename(columns={old_name: new_name})

def sort_values(df, by, ascending=True):
    return df.sort_values(by=by, ascending=ascending)

def sort_index(df, ascending=True):
    return df.sort_index(ascending=ascending)

def reset_index(df, drop=False):
    return df.reset_index(drop=drop)

def set_index(df, column_name):
    return df.set_index(column_name)

def fill_missing_values(df, value):
    return df.fillna(value)

def drop_missing_values(df):
    return df.dropna()

def replace_values(df, to_replace, value):
    return df.replace(to_replace, value)

def apply_function(df, func):
    return df.apply(func)

def applymap_function(df, func):
    return df.applymap(func)


# Aggregation and Grouping
def group_by(df, by):
    return df.groupby(by)

def aggregate(df, func):
    return df.agg(func)

def calculate_mean(df):
    return df.mean()

def calculate_sum(df):
    return df.sum()

def calculate_count(df):
    return df.count()


# Merging and Joining
def merge_dataframes(df1, df2, on, how='inner'):
    return pd.merge(df1, df2, on=on, how=how)

def join_dataframes(df1, df2, on=None, how='left'):
    return df1.join(df2, on=on, how=how)

def concatenate_dataframes(dfs, axis=0):
    return pd.concat(dfs, axis=axis)


# Outpuode
def to_csv(df, file_name):
    df.to_csv(file_name, index=False)

def to_excel(df, file_name):
    df.to_excel(file_name, index=False)

def to_sql(df, table_name, con):
    df.to_sql(table_name, con)

def to_json(df, file_name):
    df.to_json(file_name)


# Ignore Rows if Less than X%
def ignore_rows(df, threshold, axis=0):
    return df.dropna(thresh=int(threshold * len(df)), axis=axis)


# Visualization
# Correlation
def plot_correlation(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()


# Plotting with Matplotlib
def line_plot(df, x, y):
    df.plot.line(x=x, y=y)
    plt.show()

def scatter_plot(df, x, y):
    df.plot.scatter(x=x, y=y)
    plt.show()

def bar_plot(df, x, y):
    df.plot.bar(x=x, y=y)
    plt.show()

def histogram(df, column):
    df[column].plot.hist()
    plt.show()

def box_plot(df, column):
    df.boxplot(column=column)
    plt.show()

def pie_chart(df, column):
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.show()


# Plotting with Seaborn
def heatmap(df):
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()


# Handle Outliers
def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]


# Scalingde
def normalize_data(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def standardize_data(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# Dimensionality Reduction
def apply_pca(df, n_components):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    return pd.DataFrame(principal_components)


# Feature Selection
def select_k_best_features(df, target, k):
    X = df.drop(columns=[target])
    y = df[target]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return df[selected_features]


# Handle Duplicate Values
def drop_duplicates(df):
    return df.drop_duplicates()


# Split Data (Train Test)
def split_data(df, target, test_size=0.2):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size)


# Handle Imbalanced Data
def oversample_data(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def undersample_data(X, y):
    rus = RandomUnderSampler()
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

def hybrid_sampling(X, y):
    smotetomek = SMOTETomek()
    X_res, y_res = smotetomek.fit_resample(X, y)
    return X_res, y_res


# Sampling Data
def sample_data(df, n):
    return df.sample(n)


# Encodinge
def one_hot_encode(df, column):
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[[column]]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
    return df.join(encoded_df).drop(columns=[column])

def ordinal_encode(df, column):
    encoder = OrdinalEncoder()
    df[column] = encoder.fit_transform(df[[column]])
    return df

def label_encode(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df