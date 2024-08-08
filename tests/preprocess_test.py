import unittest
import weedout.preprocess as weedout

class TestWeedoutPreprocess(unittest.TestCase):
    def test_wrongpath(self):
        file_path = 'non_existent_file.csv'
        target_name = 'target_name'
        unscale_columns = []
        
        with self.assertRaises(Exception):
            weedout.initial_check_dt(file_path, target_name, unscale_columns)

if __name__ == '__main__':
    unittest.main()
