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
from unittest.mock import patch
import os
from math import sqrt
import pint
import pandas as pd
import design_tools


class TestDesignTools(unittest.TestCase):
    """
    Unit tests for detonation tube design tools
    """
    def test_read_flange_csv(self):
        """
        Tests the read_flange_csv() function, which reads flange pressure
        limits as a function of temperature for various flange classes.

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

    def test_collect_tube_materials(self):
        """
        Tests the collect_tube_materials() function, which reads in available
        materials and returns a dictionary with materials as keys and their
        appropriate ASME B16.5 material groups as values

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

    def test_get_flange_class(self):
        """
        Tests the get_flange_class() function, which takes in a temperature,
        pressure, and a string for the desired flange material and returns the
        minimum required flange class. Dataframes are assumed to be good due to
        unit testing of their import function, read_flange_csv().

        Conditions tested:
            - Function returns expected value within P, T limits
            - Proper error handling when temperature units are bad
            - Proper error handling when pressure units are bad
            - Proper error handling when temperature is outside allowable range
            - Proper error handling when pressure is outside allowable range
            - Proper error handling when temperature is not a pint quantity
            - Proper error handling when pressure is not a pint quantity
        """
        # ----------------------------INPUT TESTING----------------------------
        # check for error handling with non-string material
        # check for error handling with bad material

        # incorporate units with pint
        ureg = pint.UnitRegistry()
        quant = ureg.Quantity

        # set temperatures
        temp_low = quant(-100, ureg.degC)       # temperature too low
        temp_high = quant(500, ureg.degC)       # temperature too high
        temp_good = quant(350, ureg.degC)       # temperature and units good
        temp_badunits = quant(350, ureg.C)      # temperature good, units bad

        # set pressures
        press_low = quant(-10, ureg.bar)        # pressure too low
        press_high = quant(350, ureg.bar)       # pressure too high
        press_good = quant(125, ureg.bar)       # pressure and units good
        press_badunits = quant(125, ureg.barn)  # pressure good, units bad

        # pick material group and import lookup dataframe
        # note: T = 350 °C and P = 125 bar for group 2.3, class is 1500
        material = '316L'   # 316L is in group 2.3

        # check for expected value within limits
        test_class = design_tools.get_flange_class(temp_good,
                                                   press_good,
                                                   material)
        self.assertEqual(test_class, '1500')

        # check for error handling with bad temperature units
        with self.assertRaisesRegex(ValueError, 'Bad temperature units.'):
            design_tools.get_flange_class(temp_badunits,
                                          press_good,
                                          material)

        # check for error handling with bad pressure units
        with self.assertRaisesRegex(ValueError, 'Bad pressure units.'):
            design_tools.get_flange_class(temp_good,
                                          press_badunits,
                                          material)

        # check for error handling with temperature too low/high
        with self.assertRaisesRegex(ValueError, 'Temperature out of range.'):
            design_tools.get_flange_class(temp_low,
                                          press_good,
                                          material)
            design_tools.get_flange_class(temp_high,
                                          press_good,
                                          material)

        # check for error handling with pressure too low/high
        with self.assertRaisesRegex(ValueError, 'Pressure out of range.'):
            design_tools.get_flange_class(temp_good,
                                          press_low,
                                          material)
            design_tools.get_flange_class(temp_good,
                                          press_high,
                                          material)

        # check for error handling when temperature is not a pint quantity
        # input is numeric
        with self.assertWarnsRegex(UserWarning,
                                   'No temperature units. Assuming °C.'):
            design_tools.get_flange_class(10,
                                          press_good,
                                          material)
        # input is non-numeric
        with self.assertRaisesRegex(ValueError,
                                    'Non-numeric temperature input.'):
            design_tools.get_flange_class('asdf',
                                          press_good,
                                          material)

        # check for error handling when pressure is not a pint quantity
        # input is numeric
        with self.assertWarnsRegex(UserWarning,
                                   'No pressure units. Assuming bar.'):
            design_tools.get_flange_class(temp_good,
                                          10,
                                          material)
        # input is non-numeric
        with self.assertRaisesRegex(ValueError,
                                    'Non-numeric pressure input.'):
            design_tools.get_flange_class(temp_good,
                                          'asdf',
                                          material)

    def test_get_spiral_diameter(self):
        """
        Tests the get_spiral_diameter function, which takes the pipe inner
        diameter as a pint quantity, and the desired blockage ratio as a float
        and returns the diameter of the corresponding Shchelkin spiral.

        Conditions tested:
            - Good input
            - Proper handling with non-numeric pint ID
            - Proper handling and calculation with bad ID units
            - Proper handling with numeric, non-pint ID
            - Proper handling with non-numeric, non-pint ID
            - Proper handling with non-numeric blockage ratio
            - Proper handling with blockage ratio outside of 0<BR<100
        """
        # incorporate units with pint
        ureg = pint.UnitRegistry()
        quant = ureg.Quantity

        # define a pipe inner diameter and blockage ratio
        test_diameter = quant(5.76, ureg.inch)
        test_blockage_ratio = 44

        # define expected result and actual result
        expected_spiral_diameter = test_diameter / 2 * \
            (1 - sqrt(1 - test_blockage_ratio / 100.))
        result = design_tools.get_spiral_diameter(test_diameter,
                                                  test_blockage_ratio)

        # ensure good output
        self.assertEqual(expected_spiral_diameter, result.to(ureg.inch))

        # ensure proper handling with non-numeric pint item
        test_diameter_bad = quant('asdf', ureg.inch)
        with self.assertRaisesRegex(ValueError, 'ID is non-numeric quantity.'):
            design_tools.get_spiral_diameter(test_diameter_bad,
                                             test_blockage_ratio)

        # ensure proper handling with bad pint units
        test_diameter_bad = quant(70, ureg.degC)
        with self.assertRaisesRegex(ValueError, 'Bad diameter units.'):
            design_tools.get_spiral_diameter(test_diameter_bad,
                                             test_blockage_ratio)

        # ensure proper handling with numeric, non-pint diameter
        with self.assertWarnsRegex(Warning, 'No ID units, assuming inches.'):
            result = design_tools.get_spiral_diameter(test_diameter.magnitude,
                                                      test_blockage_ratio)
        self.assertEqual(result, expected_spiral_diameter)

        # ensure proper handling with non-numeric, non-pint diameter
        with self.assertRaisesRegex(ValueError,
                                    'ID is unitless and non-numeric.'):
            design_tools.get_spiral_diameter('oompa loompa',
                                             test_blockage_ratio)

        # ensure proper handling with non-numeric blockage ratio
        with self.assertRaisesRegex(ValueError, 'Non-numeric blockage ratio.'):
            design_tools.get_spiral_diameter(test_diameter, 'doompity doo')

        # ensure proper handling with blockage ratio outside allowable limits
        with self.assertRaisesRegex(ValueError,
                                    'Blockage ratio outside of 0<BR<100'):
            design_tools.get_spiral_diameter(test_diameter, -35.)
            design_tools.get_spiral_diameter(test_diameter, 0)
            design_tools.get_spiral_diameter(test_diameter, 100.)
            design_tools.get_spiral_diameter(test_diameter, 120)

    def test_check_materials(self):
        """
        Tests the check_materials function, which checks the materials_list
        csv file to make sure that each material contained within it has a
        corresponding flange ratings material group and tube stress limits.
        It relies on open(), collect_tube_materials(), and os.listdir(), so
        these functions have been replaced with fakes to facilitate testing.

        Conditions tested:
            - function runs correctly with good input
            - material group lookup fails
            - warning if pipe specs aren't welded or seamless
            - missing material
            - lack of flange or stress .csv file in lookup directory
        """
        class FakeOpen():
            """
            fake open()
            """
            def __init__(self, *args):
                """
                dummy init statement
                """
                return None

            def __enter__(self, *args):
                """
                enter statement that returns a FakeFile
                """
                return self.FakeFile()

            def __exit__(self, *args):
                """
                dummy exit statement
                """
                return None

            class FakeFile():
                """
                fake file used for FakeOpen
                """
                def readline(self):
                    """
                    fake file for use with FakeOpen()
                    """
                    return 'ASDF,thing0,thing1\n'

        def fake_collect_tube_materials(*args):
            """
            fake collect_tube_materials()
            """
            return {'thing0': 'group0', 'thing1': 'group1'}

        def fake_listdir(*args):
            """
            fake os.listdir() which should work 100%
            """
            return ['asdfflangegroup0sfh',
                    'asdfflangegroup1asd',
                    'asdfstressweldedasdg']

        # run test suite
        with patch('builtins.open', new=FakeOpen):
            with patch('design_tools.collect_tube_materials',
                       new=fake_collect_tube_materials):
                with patch('design_tools.listdir', new=fake_listdir):
                    # Test if function runs correctly with good input
                    self.assertIsNone(design_tools.check_materials())

                # Test if material group lookup fails
                def fake_listdir(*args):
                    """
                    listdir function which should fail group1llookup
                    """
                    return ['asdfflangegroup0sfh',
                            'asdfstressweldedasdg']
                with patch('design_tools.listdir', new=fake_listdir):
                    error_string = '\nmaterial group group1 not found'
                    with self.assertRaisesRegex(ValueError, error_string):
                        design_tools.check_materials()

                # Test for warning if pipe specs aren't welded or seamless
                def fake_listdir(*args):
                    """
                    listdir function which should warn about welded vs.
                    seamless
                    """
                    return ['asdfflangegroup0sfh',
                            'asdfflangegroup1asd',
                            'asdfstresswdasdg']
                with patch('design_tools.listdir', new=fake_listdir):
                    error_string = './lookup_data/' + 'asdfstresswdasdg' + \
                                   'does not indicate whether it is welded' + \
                                   ' or seamless'
                    with self.assertWarnsRegex(Warning, error_string):
                        design_tools.check_materials()

                # Test for missing material
                def fake_listdir(*args):
                    """
                    listdir function that should work 100%
                    """
                    return ['asdfflangegroup0sfh',
                            'asdfflangegroup1asd',
                            'asdfstressweldedasdg']
                with patch('design_tools.listdir', new=fake_listdir):
                    class NewFakeFile():
                        """
                        FakeFile class that should fail material lookup
                        """
                        def readline(self):
                            """
                            readline function that should fail material lookup
                            for thing1
                            """
                            return 'ASDF,thing0\n'
                    setattr(FakeOpen, 'FakeFile', NewFakeFile)
                    error_string = '\nMaterial thing1 not found in ./' + \
                                   'lookup_data/asdfstressweldedasdg'
                    with self.assertRaisesRegex(ValueError, error_string):
                        design_tools.check_materials()

                # Test for lack of flange or stress .csv file in lookup
                # directory
                def fake_listdir(*args):
                    """
                    listdir function that should result in flange/stress error
                    """
                    return ['asdgasdg']
                with patch('design_tools.listdir', new=fake_listdir):
                    error_string = 'no files containing "flange" or ' + \
                                   '"stress" found'
                    with self.assertRaisesRegex(ValueError, error_string):
                        design_tools.check_materials()


# %% perform unit tests
if __name__ == '__main__':
    unittest.main()
