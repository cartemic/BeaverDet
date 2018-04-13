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
import pint
import pandas as pd
import design_tools


class TestReadFlangeCsv(unittest.TestCase):
    """
    Tests the read_flange_csv() function, which reads flange pressure
    limits as a function of temperature for various flange classes.
    """
    def test_read_flange_csv(self):
        """
        Conditions tested:
            - proper error handling with bad .csv file name
            - imported dataframe has correct keys
            - imported dataframe has correct values and units
            - proper error handling when a pressure value is negative
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

        # incorporate units with pint
        ureg = pint.UnitRegistry()
        quant = ureg.Quantity

        # create test dataframe and write it to a .csv file
        test_dataframe = pd.DataFrame(data=[[0, 1],     # temperatures
                                            [2, 3]],    # pressures
                                      columns=['Temperature', 'Class'])
        test_dataframe.to_csv(file_location, index=False)

        # add units to test dataframe
        test_dataframe['Temperature'] = [quant(temp, ureg.degC) for temp in
                                         test_dataframe['Temperature']]
        test_dataframe['Class'] = [quant(pressure, ureg.bar) for pressure in
                                   test_dataframe['Class']]

        # read in test dataframe using read_flange_csv()
        test_result = design_tools.read_flange_csv(my_input)

        # check that all dataframe keys match
        self.assertListEqual(list(test_dataframe.keys()),
                             list(test_result.keys()))

        # flatten list of values and check that all dataframe values match
        test_dataframe_values = [item for column in test_dataframe.values
                                 for item in column]
        test_result_values = [item for column in test_result.values
                              for item in column]
        self.assertListEqual(test_dataframe_values, test_result_values)

        # ensure rejection of tabulated pressures less than zero
        with self.assertRaisesRegex(ValueError, 'Pressure less than zero.'):
            # create test dataframe and write it to a .csv file
            test_dataframe = pd.DataFrame(data=[[0, 1],     # temperatures
                                                [2, -3]],   # pressures
                                          columns=['Temperature', 'Class'])
            test_dataframe.to_csv(file_location, index=False)

            # run the test
            design_tools.read_flange_csv(my_input)

        # delete test .csv file from disk
        os.remove(file_location)


class TestCollectTubeMaterials(unittest.TestCase):
    """
    Tests the collect_tube_materials() function, which reads in available
    materials and returns a dictionary with materials as keys and their
    appropriate ASME B16.5 material groups as values
    """
    def test_collect_tube_materials(self):
        """
       Conditions tested:
            - values are imported correctly
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


class TestGetFlangeClass(unittest.TestCase):
    """
    Tests the get_flange_class() function, which takes in a temperature,
    pressure, and a dataframe of pressure-temperature limits for a given
    flange material and returns the minimum required flange class. Dataframes
    are assumed to be good due to unit testing of their import function,
    read_flange_csv().
    """
    def test_get_flange_class(self):
        """
        Conditions tested:
            - Function returns expected value within P, T limits
            - Proper error handling when temperature units are bad
            - Proper error handling when pressure units are bad
            - Proper error handling when temperature is outside allowable range
            - Proper error handling when pressure is outside allowable range
        """
        # ----------------------------INPUT TESTING----------------------------
        # incorporate units with pint
        ureg = pint.UnitRegistry()
        quant = ureg.Quantity

        # set temperatures
        temp_low = quant(-100, ureg.degC)
        temp_high = quant(500, ureg.degC)
        temp_good = quant(350, ureg.degC)

        # set pressures
        press_low = quant(-10, ureg.bar)
        press_high = quant(350, ureg.bar)
        press_good = quant(125, ureg.bar)

        # pick material group
        # note: T = 350 °C and P = 125 bar for group 2.3, class is 1500
        group = 2.3

        # check for expected value within limits
        self.assertEqual(design_tools.get_flange_class(temp_good, press_good),
                         1500)

        # check for error handling with bad temperature units
        # check for error handling with bad pressure units
        # check for error handling with temperature too low/high
        # check for error handling with pressure too low/high


# %% perform unit tests
if __name__ == '__main__':
    unittest.main()
