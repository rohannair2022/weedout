import unittest
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


if __name__ == '__main__':
    unittest.main()
