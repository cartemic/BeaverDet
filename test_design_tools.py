# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for design_tools.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""


import unittest
import os
import pandas as pd
import design_tools


class TestReadFlangeCsv(unittest.TestCase):
    """
    tests the read_flange_csv() function, which reads flange pressure
    limits as a function of temperature for various flange classes.
    """
    def test_read_flange_csv(self):
        """
        tests read_flange_csv (no sub-functions)
        """
        # ----------------------------INPUT TESTING----------------------------
        # ensure proper handling when a bad material group is requested

        # provide a bad input and the error that should result from it
        my_input = 'batman'
        bad_output = my_input + ' is not a valid group'

        # ensure the error is handled properly
        with self.assertRaisesRegex(ValueError, bad_output):
            design_tools.read_flange_csv(my_input)

        # ----------------------------OUTPUT TESTING---------------------------
        # ensure proper output when a good material group is requested by
        # comparing output to a known result

        # file information
        my_input = 'testfile'
        file_directory = './lookup_data/'
        file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
        file_location = file_directory + file_name

        # create test dataframe and write it to a .csv file
        test_dataframe = pd.DataFrame(data=[[0, 1], [2, 3]],
                                      columns=['Temperature', 'Class'])
        test_dataframe.to_csv(file_location, index=False)

        # read in test dataframe using read_flange_csv()
        test_result = design_tools.read_flange_csv(my_input)

        # delete test .csv file from disk
        os.remove(file_location)

        # check that dataframes are equivalent
        self.assertIsNone(pd.testing.assert_frame_equal(test_dataframe,
                                                        test_result))

        # ensure that allowable pressure values are all > 0
        print('foo')


class TestCollectTubeMaterials(unittest.TestCase):
    """
    tests the collect_tube_materials() function, which reads in available
    materials and returns a dictionary with materials as keys and their
    appropriate ASME B16.5 material groups as values
    """
    def test_collect_tube_materials(self):
        """
        tests collect_tube_materials (no sub-functions)
        """
        # ----------------------------OUTPUT TESTING---------------------------
        # ensure correctness by comparing to a pandas dataframe reading the
        # same file

        # file information
        file_directory = './lookup_data/'
        file_name = 'materials_list.csv'
        file_location = file_directory + file_name

        # load data into a test dataframe
        test_dataframe = pd.read_csv(file_location)

        # load data into a dictionary using collect_tube_materials()
        test_output = design_tools.collect_tube_materials()

        # collect keys and values from dataframe that should correspond to
        # those of the dictionary
        keys_from_dataframe = test_dataframe[test_dataframe.keys()[0]]
        values_from_dataframe = test_dataframe[test_dataframe.keys()[1]]

        for i, key in enumerate(keys_from_dataframe):
            # make sure each set of values are approximately equal
            # NOTE: this uses almost equal because of floating point errors
            self.assertAlmostEqual(float(test_output[key]),
                                   float(values_from_dataframe[i]),
                                   places=7)


# %% perform unit tests
if __name__ == '__main__':
    unittest.main()
