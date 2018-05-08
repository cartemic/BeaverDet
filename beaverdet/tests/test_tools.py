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
        - non-float values are properly ignored
        - proper error handling when a pressure value is negative
    """
    # ----------------------------INPUT TESTING----------------------------
    # ensure proper handling when a bad material group is requested

    # provide a bad input and the error that should result from it
    my_input = 'batman'
    bad_output = my_input + ' is not a valid group'

    # ensure the error is handled properly
    with pytest.raises(ValueError, match=bad_output):
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
    with pytest.raises(ValueError, match='Pressure less than zero.'):
        # create test dataframe and write it to a .csv file
        test_dataframe = pd.DataFrame(data=[[0, 1],     # temperatures
                                            [2, -3]],   # pressures
                                      columns=['Temperature', 'Class'])
        test_dataframe.to_csv(file_location, index=False)

        # run the test
        tools.get_flange_limits_from_csv(my_input)

    # create .csv to test non-numeric pressure and temperature values
    test_temperatures = [9, 's', 'd']
    test_pressures = ['a', 3, 'f']
    test_dataframe = pd.DataFrame({'Temperature': test_temperatures,
                                   'Pressure': test_pressures})
    test_dataframe.to_csv(file_location, index=False)

    # ensure non-numeric pressures and temperatures are zeroed out
    test_limits = tools.get_flange_limits_from_csv(my_input)
    for index, my_temperature in enumerate(test_temperatures):
        if isinstance(my_temperature, str):
            assert test_limits.Temperature[index].magnitude == 0

    # delete test .csv file from disk
    os.remove(file_location)


def test_lookup_flange_class():
    """
    Tests the lookup_flange_class() function, which takes in a temperature,
    pressure, and a string for the desired flange material and returns the
    minimum required flange class. Dataframes are assumed to be good due to
    unit testing of their import function, get_flange_limits_from_csv().

    Conditions tested:
        - Function returns expected value within P, T limits
        - Proper error handling when temperature is outside allowable range
        - Proper error handling when pressure is outside allowable range
        - Proper error handling when desired material isn't in database
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

    # set pressures for various tests. Values are selected to create desired
    # error conditions based on known information from 316L P-T curves.
    press_low = quant(-10, ureg.bar)        # pressure too low
    press_high = quant(350, ureg.bar)       # pressure too high
    press_good = quant(125, ureg.bar)       # pressure and units good

    # pick material group and import lookup dataframe
    # note: T = 350 °C and P = 125 bar for group 2.3, class should be 1500
    material = '316L'   # 316L is in group 2.3

    # check for expected value within limits
    test_class = tools.lookup_flange_class(temp_good, press_good, material)
    assert test_class == '1500'

    # check for error handling with temperature too low/high
    test_temperatures = [temp_low, temp_high]
    for temperature in test_temperatures:
        with pytest.raises(ValueError, match='Temperature out of range.'):
            tools.lookup_flange_class(temperature, press_good, material)

    # check for error handling with pressure too low/high
    test_pressures = [press_low, press_high]
    for pressure in test_pressures:
        with pytest.raises(ValueError, match='Pressure out of range.'):
            tools.lookup_flange_class(temp_good, pressure, material)

    # check for error handling when material isn't in database
    with pytest.raises(ValueError, match='Desired material not in database.'):
        tools.lookup_flange_class(temp_good, press_good, 'unobtainium')

    # check for error handling with non-string material
    bad_materials = [0, 3.14, -7]
    for bad_material in bad_materials:
        with pytest.raises(ValueError,
                           match='Desired material non-string input.'):
            tools.lookup_flange_class(temp_good, press_good, bad_material)


def test_calculate_spiral_diameter():
    """
    Tests the calculate_spiral_diameter function, which takes the pipe inner
    diameter as a pint quantity, and the desired blockage ratio as a float
    and returns the diameter of the corresponding Shchelkin spiral.

    Conditions tested: ADD ZERO DIAMETER CASE
        - Good input
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

    # ensure proper handling with non-numeric blockage ratio
    with pytest.raises(ValueError, match='Non-numeric blockage ratio.'):
        tools.calculate_spiral_diameter(test_diameter, 'doompity doo')

    # ensure proper handling with blockage ratio outside allowable limits
    bad_blockage_ratios = [-35.124, 0, 100, 120.34]
    for ratio in bad_blockage_ratios:
        with pytest.raises(ValueError,
                           match='Blockage ratio outside of 0<BR<100'):
            tools.calculate_spiral_diameter(test_diameter, ratio)


def test_calculate_blockage_ratio():
    """
    Tests the get_blockage_ratio function, which takes arguments of det tube
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
    test_result = tools.calculate_blockage_ratio(test_tube_diameter,
                                                 test_blockage_diameter)
    assert (test_result - hand_calc_blockage_ratio) < 1e-8
    test_result = tools.calculate_blockage_ratio(test_tube_diameter,
                                                 quant(0, ureg.inch))
    assert (test_result - hand_calc_blockage_ratio) < 1e-8

    # check for correct handling when blockage diameter >= tube diameter
    with pytest.raises(ValueError,
                       match='blockage diameter >= tube diameter'):
        tools.calculate_blockage_ratio(test_blockage_diameter,
                                       test_tube_diameter)


def test_calculate_window_sf():
    """
    Tests the calculate_window_sf function, which calculates the factor of
    safety for a viewing window.

    Conditions tested:
        - good input (all potential errors are handled by
            accessories.check_pint_quantity)
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    width = quant(50, ureg.mm).to(ureg.inch)
    length = quant(20, ureg.mm).to(ureg.mile)
    pressure = quant(1, ureg.atm).to(ureg.torr)
    thickness = quant(1.2, ureg.mm).to(ureg.furlong)
    rupture_modulus = quant(5300, ureg.psi).to(ureg.mmHg)
    desired_safety_factor = 4

    test_sf = tools.calculate_window_sf(
        length,
        width,
        thickness,
        pressure,
        rupture_modulus
    )

    assert abs(test_sf - desired_safety_factor) / test_sf < 0.01


def test_calculate_window_thk():
    """
    Tests the calculate_window_thk function, which calculates the thickness of
    a viewing window.

    Conditions tested:
        - safety factor < 1
        - non-numeric safety factor
        - good input
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    width = quant(50, ureg.mm).to(ureg.inch)
    length = quant(20, ureg.mm).to(ureg.mile)
    pressure = quant(1, ureg.atm).to(ureg.torr)
    rupture_modulus = quant(5300, ureg.psi).to(ureg.mmHg)
    desired_thickness = quant(1.2, ureg.mm)

    # safety factor < 1
    safety_factor = [0.25, -7]
    for factor in safety_factor:
        with pytest.raises(ValueError, match='Window safety factor < 1'):
            tools.calculate_window_thk(
                length,
                width,
                factor,
                pressure,
                rupture_modulus
            )

    # non-numeric safety factor
    safety_factor = 'BruceCampbell'
    with pytest.raises(TypeError, match='Non-numeric window safety factor'):
        tools.calculate_window_thk(
            length,
            width,
            safety_factor,
            pressure,
            rupture_modulus
        )

    # good input
    safety_factor = 4
    test_thickness = tools.calculate_window_thk(
        length,
        width,
        safety_factor,
        pressure,
        rupture_modulus
    )
    test_thickness = test_thickness.to(desired_thickness.units).magnitude
    desired_thickness = desired_thickness.magnitude
    assert abs(test_thickness - desired_thickness) / test_thickness < 0.01


def test_get_pipe_dlf():
    """
    Tests get_pipe_dlf

    Conditions tested:
        - good input
            * load factor is 1
            * load factor is 2
            * load factor is 4
        - plus_or_minus outside of (0, 1)
        - pipe material not in materials list
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # good input
    pipe_material = '316L'
    pipe_schedule = '80'
    nominal_pipe_size = '6'
    # from hand calcs, critical velocity is 1457.44 m/s, giving upper and lower
    # bounds of 1603.188 and 1311.700 m/s
    cj_speeds = [
        quant(1200, 'm/s'),     # DLF 1
        quant(1311, 'm/s'),     # DLF 1
        quant(1312, 'm/s'),     # DLF 4
        quant(1400, 'm/s'),     # DLF 4
        quant(1603, 'm/s'),     # DLF 4
        quant(1604, 'm/s'),     # DLF 2
        quant(2000, 'm/s')      # DLF 2
    ]
    expected_dlf = [1, 1, 4, 4, 4, 2, 2]
    for cj_speed, dlf in zip(cj_speeds, expected_dlf):
        test_dlf = tools.get_pipe_dlf(
            pipe_material,
            pipe_schedule,
            nominal_pipe_size,
            cj_speed
        )
        assert test_dlf == dlf

    # plus_or_minus outside of (0, 1)
    cj_speed = cj_speeds[0]
    bad_plus_minus = [-1, 0, 1, 2]
    for plus_minus in bad_plus_minus:
        try:
            tools.get_pipe_dlf(
                pipe_material,
                pipe_schedule,
                nominal_pipe_size,
                cj_speed,
                plus_minus
            )
        except ValueError as err:
            assert str(err) == 'plus_or_minus factor outside of (0, 1)'

    # pipe material not in materials list
    pipe_material = 'cheese'
    with pytest.raises(
            ValueError,
            match='Pipe material not found in materials_list.csv'
    ):
        tools.get_pipe_dlf(
            pipe_material,
            pipe_schedule,
            nominal_pipe_size,
            cj_speed
        )
