import unittest
import pandas as pd
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

        self.assertEqual(['feature2','feature3','target'], df.columns.to_list())

class TestCrossSectionalImputation(unittest.TestCase):

    def test_row_impute(self):
        pre_df = pd.read_csv('tests_static/csImputeCSVFile.csv')

        df = weedout.cross_sectional_imputation(pre_df)

        self.assertEqual([3.0, 5.6, 'sad', 1.0], df.iloc[2].tolist())


class TestTimeSeriesImputation(unittest.TestCase):

    def test_row_impute(self):
        pre_df = pd.read_csv('tests_static/tsImputeCSVFile.csv')

        df = weedout.time_series_imputation(pre_df)

        self.assertEqual([4.0,6.7,'sad',1.0], df.iloc[4].tolist())
    
    def test_missingnum_firstrow(self):

        with self.assertRaises(Exception) as context:
            pre_df = pd.read_csv('tests_static/tsImputeMissingCSVFile.csv')
            weedout.time_series_imputation(pre_df)

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


if __name__ == '__main__':
    unittest.main()
