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
import design_tools
import pandas as pd
import os


class TestReadFlangeCsvInput(unittest.TestCase):
    def test_read_flange_csv(self):
        # tests the read_flange_csv function, which reads flange pressure
        # limits as a function of temperature for various flange classes.
        # ---------------------------------------------------------------------
        # ensure proper handling when a bad material group is requested
        my_input = 'batman'
        bad_output = my_input + ' is not a valid group'
        with self.assertRaisesRegex(ValueError, bad_output):
            design_tools.read_flange_csv(my_input)


class TestReadFlangeCsvOutput(unittest.TestCase):
    def test_read_flange_csv(self):
        # tests the read_flange_csv function, which reads flange pressure
        # limits as a function of temperature for various flange classes.
        # ---------------------------------------------------------------------
        # ensure proper output when a good material group is requested by
        # comparing output to a known result
        my_input = 'testfile'
        file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
        test_dataframe = pd.DataFrame(data=[[0, 1], [2, 3]],
                                      columns=['Temperature', 'Class'])
        test_dataframe.to_csv(file_name, index=False)  # write test to csv
        test_result = design_tools.read_flange_csv(my_input)
        os.remove(file_name)  # delete test csv
        self.assertIsNone(pd.testing.assert_frame_equal(test_dataframe,
                                                        test_result))


class TestCollectTubeMaterials(unittest.TestCase):
    def test_collect_tube_materials(self):
        # tests the collect_tube_materials function, which reads in available
        # materials and returns a dictionary with materials as keys and their
        # appropriate ASME B16.5 material groups as values
        # ---------------------------------------------------------------------
        # ensure correctness by comparing to a pandas dataframe reading the
        # same file
        file_name = 'materials_list.csv'
        test_dataframe = pd.read_csv(file_name)
        test_output = design_tools.collect_tube_materials()
        keys_from_dataframe = test_dataframe[test_dataframe.keys()[0]]
        values_from_dataframe = test_dataframe[test_dataframe.keys()[1]]
        for i, key in enumerate(keys_from_dataframe):
            self.assertAlmostEqual(float(test_output[key]),
                                   float(values_from_dataframe[i]),
                                   places=7)


# %% perform unit tests
if __name__ == '__main__':
    unittest.main()
