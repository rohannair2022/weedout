import unittest
import pandas as pd
import numpy as np
import weedout.preprocess as weedout

class TestInitialCheckDt(unittest.TestCase):
    def test_wrong_path(self):
        file_path = 'non_existent_file.csv'
        target_name = 'target_name'
        unscale_columns = []
        
        with self.assertRaises(Exception) as context:
            weedout.initial_check_dt(file_path, target_name, unscale_columns)
        
        self.assertEqual(str(context.exception), f"An error occurred: [Errno 2] No such file or directory: '{file_path}'") 
    
    def test_wrong_targetname(self):
        file_path = 'tests_static/regularCSVFile.csv'
        target_name = 'non_existent'
        unscale_columns = []

        with self.assertRaises(Exception) as context:
            weedout.initial_check_dt(file_path, target_name, unscale_columns)

        self.assertEqual(str(context.exception), f"The target variable '{target_name}' is missing in the DataFrame.") 
    
    def test_duplicate_columns(self):
        file_path = 'tests_static/duplicateCSVFile.csv'
        target_name = 'target'
        unscale_columns = []

        with self.assertRaises(Exception) as context:
            weedout.initial_check_dt(file_path, target_name, unscale_columns)

        self.assertEqual(str(context.exception), "An error occurred: ('Duplicate columns exist in the file:', ['feature2'])") 

    def test_column_drop(self):
        file_path = 'tests_static/regularCSVFile.csv'
        target_name = 'target'
        unscale_columns = ['feature1']

        df = weedout.initial_check_dt(file_path, target_name, unscale_columns)

        self.assertEqual(['id','feature2','feature3','target'], df.columns.to_list())

class TestCrossSectionalImputation(unittest.TestCase):

    def test_row_impute(self):
        pre_df = pd.read_csv('tests_static/csImputeCSVFile.csv')
        target_name = 'target'

        df = weedout.cross_sectional_imputation(pre_df, target_name)

        self.assertEqual([3.0, 5.6, 'sad', 0.0], df.iloc[2].tolist())


class TestTimeSeriesImputation(unittest.TestCase):

    def test_row_impute(self):
        pre_df = pd.read_csv('tests_static/tsImputeCSVFile.csv')

        target_name = 'target'

        df = weedout.time_series_imputation(pre_df, target_name)

        self.assertEqual([4.0,6.7,'sad',1.0], df.iloc[4].tolist())
    
    def test_missingnum_firstrow(self):
        target_name = 'target'

        with self.assertRaises(Exception) as context:
            pre_df = pd.read_csv('tests_static/tsImputeMissingCSVFile.csv')
            weedout.time_series_imputation(pre_df, target_name)

        self.assertEqual(str(context.exception), "The first row cannot have a numeric null value.") 

class TestHandleBalanceDataset(unittest.TestCase):

    def test_oversampling(self):

        pre_df = pd.read_csv('tests_static/balanceCSVFile.csv')
        target_variable='target'
        strategy = 'oversampling'

        def min_and_max_ratio(df, target):
            max_count = df[target].value_counts().max()
            min_count = df[target].value_counts().min()
            return min_count/max_count
        
        ratio_minmax_pre = min_and_max_ratio(pre_df, target_variable)
        df = weedout.handle_imbalanced_data(pre_df, target_variable, strategy)
        ratio_minmax_post = min_and_max_ratio(df, target_variable)
        self.assertGreater(ratio_minmax_post, ratio_minmax_pre)
    
    def test_undersampling(self):

        pre_df = pd.read_csv('tests_static/balanceCSVFile.csv')
        target_variable='target'
        strategy = 'undersampling'

        def min_and_max_ratio(df, target):
            max_count = df[target].value_counts().max()
            min_count = df[target].value_counts().min()
            return min_count/max_count
        
        ratio_minmax_pre = min_and_max_ratio(pre_df, target_variable)
        df = weedout.handle_imbalanced_data(pre_df, target_variable, strategy)
        ratio_minmax_post = min_and_max_ratio(df, target_variable)
        self.assertGreater(ratio_minmax_post, ratio_minmax_pre)
    
    def test_SMOTE(self):

        pre_df = pd.read_csv('tests_static/balanceCSVFile.csv')
        target_variable='target'
        strategy = 'smote'

        def min_and_max_ratio(df, target):
            max_count = df[target].value_counts().max()
            min_count = df[target].value_counts().min()
            return min_count/max_count
        
        ratio_minmax_pre = min_and_max_ratio(pre_df, target_variable)
        df = weedout.handle_imbalanced_data(pre_df, target_variable, strategy)
        ratio_minmax_post = min_and_max_ratio(df, target_variable)
        self.assertGreater(ratio_minmax_post, ratio_minmax_pre)


class TestRemoveOutliers(unittest.TestCase):

    def test_remove_outliers(self):
         # Load the data
        df1 = pd.read_csv('tests_static/outlierCSVFile.csv')
        df2 = weedout.remove_outliers(df1)

        # Merge DataFrames with indicator to find differing rows
        diff_df = df1.merge(df2, how='outer', indicator=True)
        rows_in_df1_not_in_df2 = diff_df[diff_df['_merge'] == 'left_only']

        # Define the rows to check
        rows_to_check = pd.DataFrame({
            'Name': ['Frank', 'Jack'],
            'Age': [28, 31],
            'Height': [400, 178],
            'Weight': [200, 900]
        })

        # Select relevant columns for comparison
        columns_to_compare = ['Name', 'Age', 'Height', 'Weight']
        rows_in_df1_not_in_df2 = rows_in_df1_not_in_df2[columns_to_compare]
        rows_to_check = rows_to_check[columns_to_compare]

        # Reset index for proper comparison
        rows_in_df1_not_in_df2 = rows_in_df1_not_in_df2.reset_index(drop=True)
        rows_to_check = rows_to_check.reset_index(drop=True)

        # Assert that the DataFrames are equal
        pd.testing.assert_frame_equal(rows_in_df1_not_in_df2, rows_to_check)

class TestSeperateTargetColumn(unittest.TestCase):

    def test_file_type(self):
        df='tests_static/duplicateCSVFile.csv'
        target_name = 'target'

        with self.assertRaises(Exception) as context:
            weedout.separate_target_column(df, target_name)

        self.assertEqual(str(context.exception), "Data type error.Please provide the right type of dataframe.") 

    def test_wrong_targetname(self):
        df=pd.read_csv('tests_static/duplicateCSVFile.csv')
        target_name = 'non_existent'

        with self.assertRaises(Exception) as context:
            weedout.separate_target_column(df,target_name)

        self.assertEqual(str(context.exception), f"Target Column does not Exist. Please provide the right one.")

    def test_regular_split(self):
        pre_df = pd.read_csv('tests_static/regularCSVFile.csv')
        target_name = 'target'
        df, target= weedout.separate_target_column(pre_df, target_name)
        self.assertEqual(['id','feature1','feature2','feature3'], df.columns.to_list())
        self.assertEqual(['target'], target.columns.tolist() )

class TestEncoding(unittest.TestCase):

    def test_encoding_no_object(self):
        pre_df = pd.read_csv('tests_static/regularCSVFile.csv')
        df = weedout.encoding(pre_df)
        pd.testing.assert_frame_equal(pre_df,df)

    def test_onehot_encoding(self):
        pre_df = pd.read_csv('tests_static/oneHEncodeCSVFile.csv')
        df = weedout.encoding(pre_df,['feature1'])
        self.assertEqual(5, len(df.columns.to_list()))
        self.assertEqual(['feature1', 'feature2', 'target','feature3_happy', 'feature3_sad'], df.columns.to_list())
    
    def test_label_encoding(self):
        pre_df = pd.read_csv('tests_static/labelEncodeCSVFile.csv')
        df = weedout.encoding(pre_df,['feature1'])
        self.assertEqual(4, len(df.columns.to_list()))
        self.assertEqual(['feature1', 'feature2', 'feature3','target'], df.columns.to_list())
        self.assertEqual('int', df['feature3'].dtype)
        self.assertEqual(4, df['feature3'].nunique())


class TestCombine(unittest.TestCase):

    def test_combine(self):
        pre_df = pd.read_csv('tests_static/regularCSVFile.csv')
        features, target = weedout.separate_target_column(pre_df, 'target')
        df = weedout.combine(features, target)
        pd.testing.assert_frame_equal(pre_df, df)

    def test_combine_fail(self):
        pre_df = pd.read_csv('tests_static/regularCSVFile.csv')
        with self.assertRaises(Exception) as context:
            pre_df = pd.read_csv('tests_static/tsImputeMissingCSVFile.csv')
            weedout.combine(pre_df, "target")
        self.assertEqual(str(context.exception), f"Data type error.Please provide the right type of dataframe.")

class TestFeatureScaling(unittest.TestCase):
        
        def test_featurescaling(self):
        
            n = 200

            normal_1 = np.random.normal(size=n)
            normal_2 = np.random.normal(size=n)

            exponential = np.random.exponential(size=n)
            lognormal = np.random.lognormal(size=n)
            binary_values = np.random.randint(2, size=n)

            pre_df = pd.DataFrame({
                'Normal_1': normal_1,
                'Exponential': exponential,
                'Lognormal': lognormal,
                'Normal_2': normal_2,
                'Binary': binary_values,
            })

            df = weedout.feature_scaling(pre_df, [])

            # MinMaxScaled Columns should result in values between 0 and 1. 
            self.assertEqual(0, df['Exponential'].min())
            self.assertAlmostEqual(1, df['Exponential'].max())
            self.assertEqual(0, df['Lognormal'].min())
            self.assertAlmostEqual(1, df['Lognormal'].max())

            # Binary values should not be worked upon.
            pd.testing.assert_frame_equal(df[['Binary']], pre_df[['Binary']])


class TestPipeline(unittest.TestCase):

    def test_final_pipeline(self):

        pre_df = pd.read_csv('tests_static/regularCSVFile.csv')
        target_variable ='target'
        strategy = "smote"
        sampling = 1
        classification = 1


        df = weedout.preprocess_pipeline('tests_static/regularCSVFile.csv', target_variable, ['feature1'], ['id'], type_dataset=0, sampling=1, classfication=1, strategy_sample=strategy)
        
        self.assertEqual('object', df['id'].dtype)
        self.assertTrue(not df.isnull().values.any())

        # Test for Smote
        if classification:
            def min_and_max_ratio(df, target):
                max_count = df[target].value_counts().max()
                min_count = df[target].value_counts().min()
                return min_count/max_count
            
            ratio_minmax_pre = min_and_max_ratio(pre_df, target_variable)
            ratio_minmax_post = min_and_max_ratio(df, target_variable)
            self.assertGreater(ratio_minmax_post, ratio_minmax_pre)

        # Test for no null values:
        if sampling:
            if classification:
                self.assertEqual(0, df.isnull().sum().sum())
        

if __name__ == '__main__':
    unittest.main()
