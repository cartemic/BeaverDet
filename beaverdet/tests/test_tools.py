# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for tools.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""


import os
from math import sqrt
import pytest
import pint
import pandas as pd
from ..tube_design_tools import tools


def test_get_flange_limits_from_csv():
    """
    Tests the get_flange_limits_from_csv() function, which reads flange
    pressure limits as a function of temperature for various flange classes.

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
    with pytest.raises(ValueError, message=bad_output):
        tools.get_flange_limits_from_csv(my_input)

    # ----------------------------OUTPUT TESTING---------------------------
    # ensure proper output when a good material group is requested by
    # comparing output to a known result

    # file information
    my_input = 'testfile'
    file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'tube_design_tools', 'lookup_data')
    file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
    file_location = os.path.relpath(os.path.join(file_directory, file_name))

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

    # read in test dataframe using get_flange_limits_from_csv()
    test_result = tools.get_flange_limits_from_csv(my_input)

    # check that dataframes have the same number of keys and that they match
    assert len(test_dataframe.keys()) == len(test_result.keys())
    assert all(key1 == key2 for key1, key2 in zip(test_dataframe.keys(),
                                                  test_result.keys()))

    # flatten list of values and check that all dataframe values match
    test_dataframe_values = [item for column in test_dataframe.values
                             for item in column]
    test_result_values = [item for column in test_result.values
                          for item in column]
    assert len(test_dataframe_values) == len(test_result_values)
    assert all(val1 == val2 for val1, val2 in zip(test_dataframe_values,
                                                  test_result_values))

    # ensure rejection of tabulated pressures less than zero
    with pytest.raises(ValueError, message='Pressure less than zero.'):
        # create test dataframe and write it to a .csv file
        test_dataframe = pd.DataFrame(data=[[0, 1],     # temperatures
                                            [2, -3]],   # pressures
                                      columns=['Temperature', 'Class'])
        test_dataframe.to_csv(file_location, index=False)

        # run the test
        tools.get_flange_limits_from_csv(my_input)

    # delete test .csv file from disk
    os.remove(file_location)


def test_get_flange_class():
    """
    Tests the get_flange_class() function, which takes in a temperature,
    pressure, and a string for the desired flange material and returns the
    minimum required flange class. Dataframes are assumed to be good due to
    unit testing of their import function, get_flange_limits_from_csv().

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

    # set temperatures for various tests. Values are selected to create desired
    # error conditions based on known information from 316L P-T curves.
    temp_low = quant(-100, ureg.degC)       # temperature too low
    temp_high = quant(500, ureg.degC)       # temperature too high
    temp_good = quant(350, ureg.degC)       # temperature and units good
    temp_badunits = quant(350, ureg.C)      # temperature good, units bad

    # set pressures for various tests. Values are selected to create desired
    # error conditions based on known information from 316L P-T curves.
    press_low = quant(-10, ureg.bar)        # pressure too low
    press_high = quant(350, ureg.bar)       # pressure too high
    press_good = quant(125, ureg.bar)       # pressure and units good
    press_badunits = quant(125, ureg.barn)  # pressure good, units bad

    # pick material group and import lookup dataframe
    # note: T = 350 °C and P = 125 bar for group 2.3, class should be 1500
    material = '316L'   # 316L is in group 2.3

    # check for expected value within limits
    test_class = tools.get_flange_class(temp_good, press_good, material)
    assert test_class == '1500'

    # check for error handling with bad temperature units
    with pytest.raises(ValueError, message='Bad temperature units.'):
        tools.get_flange_class(temp_badunits, press_good, material)

    # check for error handling with bad pressure units
    with pytest.raises(ValueError, message='Bad pressure units.'):
        tools.get_flange_class(temp_good, press_badunits, material)

    # check for error handling with temperature too low/high
    with pytest.raises(ValueError, message='Temperature out of range.'):
        tools.get_flange_class(temp_low, press_good, material)
        tools.get_flange_class(temp_high, press_good, material)

    # check for error handling with pressure too low/high
    with pytest.raises(ValueError, message='Pressure out of range.'):
        tools.get_flange_class(temp_good, press_low, material)
        tools.get_flange_class(temp_good, press_high, material)

    # check for error handling when temperature is not a pint quantity
    # input is numeric
    with pytest.warns(UserWarning, match='No temperature units. Assuming °C.'):
        tools.get_flange_class(10, press_good, material)
    # input is non-numeric
    with pytest.raises(ValueError, message='Non-numeric temperature input.'):
        tools.get_flange_class('asdf', press_good, material)

    # check for error handling when pressure is not a pint quantity
    # input is numeric
    with pytest.warns(UserWarning, match='No pressure units. Assuming bar.'):
        tools.get_flange_class(temp_good, 10, material)
    # input is non-numeric
    with pytest.raises(ValueError, message='Non-numeric pressure input.'):
        tools.get_flange_class(temp_good, 'asdf', material)


def test_calculate_spiral_diameter():
    """
    Tests the calculate_spiral_diameter function, which takes the pipe inner
    diameter as a pint quantity, and the desired blockage ratio as a float
    and returns the diameter of the corresponding Shchelkin spiral.

    Conditions tested: ADD ZERO DIAMETER CASE
        - Good input
        - Proper handling with non-numeric pint ID
        - Proper handling and calculation with bad ID units
        - Proper handling with numeric, non-pint ID
        - Proper handling with non-numeric, non-pint ID
        - Proper handling with non-numeric blockage ratio
        - Proper handling with blockage ratio outside of 0<BR<100
        - Proper handling with tube of diameter 0
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
    result = tools.calculate_spiral_diameter(test_diameter,
                                             test_blockage_ratio)

    # ensure good output
    assert expected_spiral_diameter == result.to(ureg.inch)

    # ensure proper handling with non-numeric pint item
    test_diameter_bad = quant('asdf', ureg.inch)
    with pytest.raises(ValueError, message='ID is non-numeric quantity.'):
        tools.calculate_spiral_diameter(test_diameter_bad, test_blockage_ratio)

    # ensure proper handling with bad pint units
    test_diameter_bad = quant(70, ureg.degC)
    with pytest.raises(ValueError, message='Bad diameter units.'):
        tools.calculate_spiral_diameter(test_diameter_bad, test_blockage_ratio)

    # ensure proper handling with numeric, non-pint diameter
    with pytest.warns(Warning, match='No ID units, assuming inches.'):
        result = tools.calculate_spiral_diameter(test_diameter.magnitude,
                                                 test_blockage_ratio)
    assert result == expected_spiral_diameter

    # ensure proper handling with non-numeric, non-pint diameter
    with pytest.raises(ValueError, message='ID is unitless and non-numeric.'):
        tools.calculate_spiral_diameter('oompa loompa', test_blockage_ratio)

    # ensure proper handling with non-numeric blockage ratio
    with pytest.raises(ValueError, message='Non-numeric blockage ratio.'):
        tools.calculate_spiral_diameter(test_diameter, 'doompity doo')

    # ensure proper handling with blockage ratio outside allowable limits
    with pytest.raises(ValueError,
                       message='Blockage ratio outside of 0<BR<100'):
        tools.calculate_spiral_diameter(test_diameter, -35.)
        tools.calculate_spiral_diameter(test_diameter, 0)
        tools.calculate_spiral_diameter(test_diameter, 100.)
        tools.calculate_spiral_diameter(test_diameter, 120)


def test_get_blockage_ratio():
    """
    Tests the get_blockage ratio function, which takes arguments of det tube
    inner diameter and spiral blockage diameter.

    Conditions tested:
        - good input (both zero and nonzero)
        - units are mismatched
        - non-pint blockage diameter
        - non-numeric pint blockage diameter
        - blockage diameter with bad units
        - non-pint tube diameter
        - non-numeric pint tube diameter
        - tube diameter with bad units
        - blockage diameter < 0
        - blockage diameter >= tube diameter
        - tube diameter < 0
    """
    # incorporate units with pint
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # known good results from hand-calcs
    test_tube_diameter = quant(3.438, ureg.inch)
    test_blockage_diameter = quant(7./16., ureg.inch)
    hand_calc_blockage_ratio = 0.444242326717679 * 100

    # check for expected result with good input
    test_result = tools.get_blockage_ratio(test_tube_diameter,
                                           test_blockage_diameter)
    assert (test_result - hand_calc_blockage_ratio) < 1e-8
    test_result = tools.get_blockage_ratio(test_tube_diameter,
                                           quant(0, ureg.inch))
    assert (test_result - hand_calc_blockage_ratio) < 1e-8

    # check for correct result when units are mismatched
    test_result = tools.get_blockage_ratio(test_tube_diameter.to(ureg.meter),
                                           test_blockage_diameter)
    assert (test_result - hand_calc_blockage_ratio) < 1e-8

    # check for correct handling of non-pint blockage diameter
    with pytest.raises(ValueError,
                       message='blockage diameter is not a pint quantity'):
        tools.get_blockage_ratio(test_tube_diameter, 'adsg')
        tools.get_blockage_ratio(test_tube_diameter, 7)
        tools.get_blockage_ratio(test_tube_diameter, 2.125)

    # check for correct handling of non-numeric pint blockage diameter
    with pytest.raises(ValueError, message='blockage diameter is non-numeric'):
        tools.get_blockage_ratio(test_tube_diameter, quant('asd', ureg.inch))

    # check for correct handling of blockage diameter with bad units
    with pytest.raises(ValueError, message='blockage diameter has bad units'):
        tools.get_blockage_ratio(test_tube_diameter, quant(23, ureg.degC))

    # check for correct handling of non-pint tube diameter
    with pytest.raises(ValueError,
                       message='tube diameter is not a pint quantity'):
        tools.get_blockage_ratio('adsg', test_blockage_diameter)
        tools.get_blockage_ratio(7, test_blockage_diameter)
        tools.get_blockage_ratio(2.125, test_blockage_diameter)

    # check for correct handling of non-numeric pint tube diameter
    with pytest.raises(ValueError, message='tube diameter is non-numeric'):
        tools.get_blockage_ratio(quant('asd', ureg.inch),
                                 test_blockage_diameter)

    # check for correct handling of tube diameter with bad units
    with pytest.raises(ValueError, message='tube diameter has bad units'):
        tools.get_blockage_ratio(quant(23, ureg.degC), test_blockage_diameter)

    # check for correct handling when blockage diameter < 0
    with pytest.raises(ValueError, message='blockage diameter < 0'):
        tools.get_blockage_ratio(test_tube_diameter, -test_blockage_diameter)

    # check for correct handling when blockage diameter >= tube diameter
    with pytest.raises(ValueError,
                       message='blockage diameter >= tube diameter'):
        tools.get_blockage_ratio(test_blockage_diameter, test_tube_diameter)

    # check for correct handling when tube diameter <= 0
    with pytest.raises(ValueError, message='tube diameter <= 0'):
        tools.get_blockage_ratio(-test_tube_diameter, test_blockage_diameter)
