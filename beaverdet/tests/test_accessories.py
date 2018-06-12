# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for accessories.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import os
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
import pint
from ..tube_design_tools import accessories



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
        accessories.get_flange_limits_from_csv(my_input)

    # ----------------------------OUTPUT TESTING---------------------------
    # ensure proper output when a good material group is requested by
    # comparing output to a known result

    # file information
    my_input = 'testfile'
    file_directory = os.path.join(os.path.dirname(os.path.relpath(__file__)),
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
    test_result = accessories.get_flange_limits_from_csv(my_input)

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
        accessories.get_flange_limits_from_csv(my_input)

    # create .csv to test non-numeric pressure and temperature values
    test_temperatures = [9, 's', 'd']
    test_pressures = ['a', 3, 'f']
    test_dataframe = pd.DataFrame({'Temperature': test_temperatures,
                                   'Pressure': test_pressures})
    test_dataframe.to_csv(file_location, index=False)

    # ensure non-numeric pressures and temperatures are zeroed out
    test_limits = accessories.get_flange_limits_from_csv(my_input)
    for index, my_temperature in enumerate(test_temperatures):
        if isinstance(my_temperature, str):
            assert test_limits.Temperature[index].magnitude == 0

    # delete test .csv file from disk
    os.remove(file_location)


def test_check_materials():
    """
    Tests the check_materials function, which checks the materials_list
    csv file to make sure that each material contained within it has a
    corresponding flange ratings material group and tube stress limits.
    It relies on open(), get_material_groups(), and os.listdir(), so
    these functions have been replaced with fakes to facilitate testing.

    Conditions tested:
        - function runs correctly with good input
        - material group lookup fails
        - warning if pipe specs aren't welded or seamless
        - missing material
        - lack of flange or stress .csv file in lookup directory
    """
    file_directory = os.path.join(
        os.path.dirname(
            os.path.relpath(__file__)
        ),
        '..',
        'tube_design_tools',
        'lookup_data'
    )

    class FakeOpen:
        """
        fake open()
        """
        def __init__(self, *_):
            """
            dummy init statement
            """

        def __enter__(self, *_):
            """
            enter statement that returns a FakeFile
            """
            return self.FakeFile()

        def __exit__(self, *_):
            """
            dummy exit statement
            """
            return None

        class FakeFile:
            """
            fake file used for FakeOpen
            """
            @staticmethod
            def readline(*_):
                """
                fake file for use with FakeOpen()
                """
                return 'ASDF,thing0,thing1\n'

    def fake_get_material_groups(*_):
        """
        fake get_material_groups()
        """
        return {'thing0': 'group0', 'thing1': 'group1'}

    def fake_listdir(*_):
        """
        fake os.listdir() which should work 100%
        """
        return ['asdfflangegroup0sfh',
                'asdfflangegroup1asd',
                'asdfstressweldedasdg']

    # run test suite
    with patch('builtins.open', new=FakeOpen):
        patched_module = __name__.split('.')[0] + \
            '.tube_design_tools.accessories.' + \
            'get_material_groups'
        with patch(patched_module,
                   new=fake_get_material_groups):
            with patch('os.listdir', new=fake_listdir):
                # Test if function runs correctly with good input
                assert accessories.check_materials() is None

            # Test if material group lookup fails
            def fake_listdir(*_):
                """
                listdir function which should fail group1llookup
                """
                return ['asdfflangegroup0sfh',
                        'asdfstressweldedasdg']
            with patch('os.listdir', new=fake_listdir):
                error_string = '\nmaterial group group1 not found'
                with pytest.raises(ValueError, message=error_string):
                    accessories.check_materials()

            # Test for warning if pipe specs aren't welded or seamless
            def fake_listdir(*_):
                """
                listdir function which should warn about welded vs.
                seamless
                """
                return ['asdfflangegroup0sfh',
                        'asdfflangegroup1asd',
                        'asdfstresswdasdg']
            with patch('os.listdir', new=fake_listdir):
                error_string = 'asdfstresswdasdg' + \
                               'does not indicate whether it is welded' + \
                               ' or seamless'
                with pytest.warns(Warning, match=error_string):
                    accessories.check_materials()

            # Test for missing material
            def fake_listdir(*_):
                """
                listdir function that should work 100%
                """
                return ['asdfflangegroup0sfh',
                        'asdfflangegroup1asd',
                        'asdfstressweldedasdg']
            with patch('os.listdir', new=fake_listdir):
                class NewFakeFile:
                    """
                    FakeFile class that should fail material lookup
                    """
                    @staticmethod
                    def readline(*_):
                        """
                        readline function that should fail material lookup
                        for thing1
                        """
                        return 'ASDF,thing0\n'
                setattr(FakeOpen, 'FakeFile', NewFakeFile)
                error_string = '\nMaterial thing1 not found in ' + \
                               os.path.join(file_directory,
                                            'asdfstressweldedasdg')
                with pytest.raises(ValueError, message=error_string):
                    accessories.check_materials()

            # Test for lack of flange or stress .csv file in lookup
            # directory
            def fake_listdir(*_):
                """
                listdir function that should result in flange/stress error
                """
                return ['asdgasdg']
            with patch('os.listdir', new=fake_listdir):
                error_string = 'no files containing "flange" or ' + \
                               '"stress" found'
                with pytest.raises(ValueError, message=error_string):
                    accessories.check_materials()


def test_collect_tube_materials():
    """
    Tests collect_tube_materials

    Conditions tested:
        - file doesn't exist
        - file is empty
        - good input
    """
    file_directory = os.path.join(
        os.path.dirname(
            os.path.relpath(__file__)
        ),
        '..',
        'tube_design_tools',
        'lookup_data'
    )
    file_name = 'materials_list.csv'
    file_location = os.path.relpath(
        os.path.join(
            file_directory,
            file_name
        )
    )
    new_file_name = 'materials_list_original'
    new_file_location = os.path.relpath(
        os.path.join(
            file_directory,
            new_file_name
        )
    )

    # rename materials_list.csv for testing
    os.rename(file_location, new_file_location)

    # file doesn't exist
    with pytest.raises(
        ValueError,
        match=file_name+' does not exist'
    ):
        accessories.collect_tube_materials()

    # file is empty
    open(file_location, 'a').close()
    with pytest.raises(
        ValueError,
        match=file_name+' is empty'
    ):
        accessories.collect_tube_materials()

    # good input
    fake_columns = [
        'Grade', 'Group', 'ElasticModulus', 'Density', 'Poisson'
    ]
    fake_data = [
        304, 2.1, 200, 7.8, 0.28
    ]
    test_dataframe = pd.DataFrame(fake_data, fake_columns).transpose()
    test_dataframe.to_csv(file_location, index=False)
    test_materials = accessories.collect_tube_materials()
    # check length
    assert test_materials.shape[1] == len(fake_data)
    # check units
    assert (test_materials.ElasticModulus[0].units.format_babel()
            ==
            'gigapascal')
    assert (test_materials.Density[0].units.format_babel()
            ==
            'gram / centimeter ** 3')
    for column, data in zip(fake_columns, fake_data):
        current_item = test_materials[column][0]
        try:
            # remove units from pint quantities
            current_item = current_item.magnitude
        except AttributeError:
            pass
        assert current_item == data

    # reinstate original materials list
    os.remove(file_location)
    os.rename(new_file_location, file_location)


def test_get_material_groups():
    """
    Tests the get_material_groups() function, which reads in available
    materials and returns a dictionary with materials as keys and their
    appropriate ASME B16.5 material groups as values

   Conditions tested:
        - file does not exist
        - file is empty
        - values are imported correctly
    """
    # file information
    file_directory = os.path.join(
        os.path.dirname(
            os.path.relpath(__file__)
        ),
        '..',
        'tube_design_tools',
        'lookup_data'
    )
    file_name = 'materials_list.csv'
    file_location = os.path.join(file_directory, file_name)

    # ----------------------------INPUT TESTING----------------------------
    # ensure proper error handling with bad inputs

    # check for error handling when file does not exist by removing the file
    # extension
    bad_location = file_location[:-4]
    os.rename(file_location, bad_location)
    with pytest.raises(ValueError, message=file_name+' does not exist'):
        accessories.get_material_groups()

    # create a blank file
    open(file_location, 'a').close()

    # check for proper error handling when file is blank
    with pytest.raises(ValueError, message=file_name+' is empty'):
        accessories.get_material_groups()

    # delete the test file and reinstate the original
    os.remove(file_location)
    os.rename(bad_location, file_location)

    # ----------------------------OUTPUT TESTING---------------------------
    # ensure correctness by comparing to a pandas dataframe reading the
    # same file

    # load data into a test dataframe
    test_dataframe = pd.read_csv(file_location)

    # load data into a dictionary using get_material_groups()
    test_output = accessories.get_material_groups()

    # collect keys and values from dataframe that should correspond to
    # those of the dictionary
    keys_from_dataframe = test_dataframe.Grade.values.astype(str)
    values_from_dataframe = test_dataframe.Group.values.astype(str)

    for index, key in enumerate(keys_from_dataframe):
        # make sure each set of values are approximately equal
        dict_value = test_output[key]
        dataframe_value = values_from_dataframe[index]
        assert dict_value == dataframe_value


def test_check_pint_quantity():
    """
    Tests the check_pint_quantity() function, which makes sure a variable is
    a numeric pint quantity of the correct dimensionality. It can also ensure
    that the magnitude of the quantity is > 0.

   Conditions tested:
        - bad dimension type
        - non-pint quantity
        - non-numeric pint quantity
        - negative magnitude
        - incorrect dimensionality
        - good input
    """

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # Bad dimension type
    bad_dimension_type = 'walrus'
    error_str = bad_dimension_type + ' not a supported dimension type'
    with pytest.raises(ValueError, match=error_str):
        accessories.check_pint_quantity(
            quant(3, ureg.degC),
            bad_dimension_type
        )

    # Non-pint quantity
    with pytest.raises(ValueError, match='Non-pint quantity'):
        accessories.check_pint_quantity(
            7,
            'length'
        )

    # Non-numeric pint quantity
    with pytest.raises(ValueError, match='Non-numeric pint quantity'):
        accessories.check_pint_quantity(
            quant('asdf', ureg.inch),
            'length'
        )

    # Negative magnitude
    with pytest.raises(ValueError, match='Input value < 0'):
        accessories.check_pint_quantity(
            quant(-4, ureg.inch),
            'length',
            ensure_positive=True
        )

    # Incorrect dimensionality
    error_str = (
            ureg.degC.dimensionality.__str__() +
            ' is not ' +
            ureg.meter.dimensionality.__str__()
    )
    try:
        accessories.check_pint_quantity(
            quant(19.2, ureg.degC),
            'length'
        )
    except ValueError as err:
        assert str(err) == error_str

    # Good input
    assert accessories.check_pint_quantity(
        quant(6.3, ureg.inch),
        'length'
    )
    assert accessories.check_pint_quantity(
        quant(-8, ureg.inch),
        'length'
    )


def test_window_sympy_solver():
    """
    Tests the check_pint_quantity() function, which makes sure a variable is
    a numeric pint quantity of the correct dimensionality. It can also ensure
    that the magnitude of the quantity is > 0.

   Conditions tested:
        - pass too few arguments
        - pass bad keyword argument
        - pass arguments that cause an imaginary answer
        - good input
    Returns
    """
    # pass too few arguments
    error_msg = 'Incorrect number of arguments sent to solver'
    args = [
        {
            'length': 5,
            'width': 4,
            'thickness': 2,
            'pressure': 2
        },
        {
            'length': 5,
            'width': 4,
            'thickness': 2,
            'pressure': 2,
            'rupture_modulus': 10,
            'safety_factor': 4
        }
    ]
    for argument_dict in args:
        with pytest.raises(ValueError, match=error_msg):
            accessories.window_sympy_solver(
                **argument_dict
            )

    # pass bad keyword argument
    kwargs = args[0]
    kwargs['pulled_pork'] = 9
    with pytest.raises(ValueError, match='Bad keyword argument:\npulled_pork'):
        accessories.window_sympy_solver(**kwargs)

    # pass arguments that cause an imaginary answer
    kwargs = {
        'length': 20,
        'width': 50,
        'safety_factor': 4,
        'pressure': -139,
        'rupture_modulus': 5300
    }
    with pytest.warns(Warning,
                      match='Window inputs resulted in imaginary solution.'):
        test_nan = accessories.window_sympy_solver(**kwargs)

    assert test_nan is np.nan

    # good input
    args = [
        {
            'length': 20,
            'width': 50,
            'safety_factor': 4,
            'pressure': 14.7,
            'rupture_modulus': 5300
        },
        {
            'length': 20,
            'width': 50,
            'thickness': 1.2,
            'pressure': 14.7,
            'rupture_modulus': 5300
        }
    ]
    good_solutions = [1.2, 4.]
    for index in range(len(args)):
        test_output = accessories.window_sympy_solver(**args[index])
        assert abs(test_output - good_solutions[index]) < 0.1


def test_calculate_laminar_flamespeed():
    """
    Tests the calculate_laminar_flamespeed function
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    initial_temperature = quant(300, 'K')
    initial_pressure = quant(1, 'atm')

    # test with bad species
    species = {
        'Wayne': 3,
        'CH4': 7,
        'Garth': 5
    }
    bad_string = 'Species not in mechanism:\nWayne\nGarth\n'
    with pytest.raises(ValueError, match=bad_string):
        accessories.calculate_laminar_flamespeed(
            initial_temperature,
            initial_pressure,
            species,
            'gri30.cti'
        )

    # test with no species
    species = {}
    with pytest.raises(ValueError, match='Empty species dictionary'):
        accessories.calculate_laminar_flamespeed(
            initial_temperature,
            initial_pressure,
            species,
            'gri30.cti'
        )

    # test with good species
    species = {
        'CH4': 0.095057034220532327,
        'O2': 0.19011406844106465,
        'N2': 0.71482889733840305
    }
    good_result = 0.39  # value approximated from Law fig. 7.7.7
    test_flamespeed = accessories.calculate_laminar_flamespeed(
        initial_temperature,
        initial_pressure,
        species,
        'gri30.cti'
    )
    assert abs(test_flamespeed.magnitude - good_result) / good_result < 0.05


def test_import_pipe_schedules():
    """
    Tests import_pipe_schedules
    """
    schedule_dataframe = accessories.import_pipe_schedules()
    good_values = [
        ('5s', '1/2', 0.065),
        ('40', '2 1/2', 0.203),
        ('XXH', '8', 0.875)
    ]
    for (schedule, size, thickness) in good_values:
        assert schedule_dataframe[schedule][size] == thickness


def test_get_available_pipe_sizes():
    """
    Tests get_available_pipe_sizes

    Conditions tested:
        - good input
        - non-existent pipe size
    """
    schedule_info = accessories.import_pipe_schedules()

    # good input
    test_sizes = accessories.get_available_pipe_sizes('40', schedule_info)
    assert isinstance(test_sizes, list)
    assert '36' in test_sizes
    assert '42' not in test_sizes

    # non-existent pipe size
    with pytest.raises(ValueError, match='Pipe class not found'):
        accessories.get_available_pipe_sizes(
            'how do you type with boxing gloves on?',
            schedule_info
        )


def test_get_pipe_dimensions():
    """
    Tests get_pipe_dimensions

    Conditions tested:
        - good input
        - bad pipe size
    """
    # good input
    dimensions = accessories.get_pipe_dimensions(
        pipe_schedule='80',
        nominal_size='6'
    )
    outer_diameter = dimensions['outer diameter']
    inner_diameter = dimensions['inner diameter']
    wall_thickness = dimensions['wall thickness']
    assert outer_diameter.magnitude - 6.625 < 1e-7
    assert inner_diameter.magnitude - 5.761 < 1e-7
    assert wall_thickness.magnitude - 0.432 < 1e-7
    assert outer_diameter.units.format_babel() == 'inch'
    assert inner_diameter.units.format_babel() == 'inch'
    assert wall_thickness.units.format_babel() == 'inch'

    # bad pipe size
    with pytest.raises(
            ValueError,
            match='Nominal size not found for given pipe schedule'
    ):
        accessories.get_pipe_dimensions(
            pipe_schedule='80',
            nominal_size='really big'
        )


def test_import_thread_specs():
    """
    Tests import_thread_specs

    Conditions tested:
        - good input
    """
    # good input
    test_size = '0-80'
    test_type = ['external', 'internal']
    test_property = 'pitch diameter max'
    test_classes = [['2A', '3A'], ['2B', '3B']]
    good_result = [[0.0514, 0.0519], [0.0542, 0.0536]]
    specifications = accessories.import_thread_specs()
    assert isinstance(specifications, dict)
    for thread, classes, expected in zip(test_type, test_classes, good_result):
        # check type
        current_frame = specifications[thread]
        assert isinstance(current_frame, pd.DataFrame)

        # check output
        for thread_class, result in zip(classes, expected):
            test_result = current_frame[test_property][test_size][thread_class]
            assert abs(test_result - result) < 1e-7


def test_get_thread_property():
    """
    Tests get_thread_property

    Conditions tested:
        - good input
        - thread_specs not a dataframe
        - non-string property
        - property not in dataframe
        - non-string thread size
        - thread size not in dataframe
        - non-string thread class
        - thread class not in dataframe
    """
    dataframes = accessories.import_thread_specs()

    '''
        thread_property,
        thread_size,
        thread_class,
        thread_specs
    '''
    # good input
    good_args = [
        [
            'pitch diameter max',
            '1/4-20',
            '2B',
            dataframes['internal']
        ],
        [
            'pitch diameter max',
            '1/4-20',
            '2A',
            dataframes['external']
        ]
    ]
    good_results = [
        0.2248,
        0.2164
    ]
    for args, result in zip(good_args, good_results):
        test_result = accessories.get_thread_property(*args)

        # type/unit check
        assert test_result.units.format_babel() == 'inch'

        # magnitude check
        assert abs(test_result.magnitude - result) < 1e-7

    # thread specs not a dataframe
    with pytest.raises(
            TypeError,
            match='thread_specs is not a pandas dataframe'
    ):
        accessories.get_thread_property(
            'pitch diameter max',
            '1/4-20',
            '2B',
            'whoa'
        )

    # non-string property
    with pytest.raises(
          TypeError,
          match='thread_property expected a string'
    ):
        accessories.get_thread_property(
            58008,
            '1/4-20',
            '2B',
            dataframes['internal']
        )

    # property not in dataframe
    bad_property = 'jello'
    bad_message = (
            'Thread property \'' +
            bad_property +
            '\' not found. Available specs: ' +
            "'" + "', '".join(dataframes['internal'].keys()) + "'"
    )
    with pytest.raises(KeyError, match=bad_message):
        accessories.get_thread_property(
            bad_property,
            '1/4-20',
            '2B',
            dataframes['internal']
        )

    # non-string thread size
    with pytest.raises(
          TypeError,
          match='thread_size expected a string'
    ):
        accessories.get_thread_property(
            'pitch diameter max',
            58008,
            '2B',
            dataframes['internal']
        )

    # thread size not in dataframe
    bad_message = (
            'Thread size \'' +
            bad_property +
            '\' not found'
    )
    with pytest.raises(KeyError, match=bad_message):
        accessories.get_thread_property(
            'pitch diameter max',
            bad_property,
            '2B',
            dataframes['internal']
        )

    # non-string thread class
    with pytest.raises(
          TypeError,
          match='thread_class expected a string'
    ):
        accessories.get_thread_property(
            'pitch diameter max',
            '1/4-20',
            58008,
            dataframes['internal']
        )

    # thread class not in dataframe
    bad_message = (
            'Thread class \'' +
            bad_property +
            '\' not found'
    )
    with pytest.raises(KeyError, match=bad_message):
        accessories.get_thread_property(
            'pitch diameter max',
            '1/4-20',
            bad_property,
            dataframes['internal']
        )


def test_get_thread_tpi():
    """
    Tests get_thread_tpi

    Conditions tested:
        - good input
    """
    assert accessories.get_thread_tpi('1/4-20') == 20


def test_equil_sound_speed():
    """
    Tests get_equil_sound_speed
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # check air at 1 atm and 20°C against ideal gas calculation
    gamma = 1.4
    rr = 8.31451
    tt = 293.15
    mm = 0.0289645
    c_ideal = np.sqrt(gamma*rr*tt/mm)

    temp = quant(20, 'degC')
    press = quant(1, 'atm')
    species = {'O2': 1, 'N2': 3.76}
    mech = 'gri30.cti'
    c_test = accessories.get_equil_sound_speed(
        temp,
        press,
        species,
        mech
    )

    assert abs(c_ideal - c_test.to('m/s').magnitude) / c_ideal <= 0.005


def test_get_pipe_stress_limits():
    material = '304'

    # known values for 304
    seamless_values = np.array([18.8, 18.8, 15.7, 14.1, 13, 12.2, 11.4, 11.3,
                                11.1, 10.8, 10.6, 10.4, 10.2, 10, 9.8, 9.5, 8.9,
                                7.7, 6.1])
    welded_values = np.array([16, 16, 13.3, 12, 11, 10.5, 9.7, 9.5, 9.4, 9.2, 9,
                              8.8, 8.7, 8.5, 8.3, 8.1, 7.6, 6.5, 5.4])

    test_limits = accessories.get_pipe_stress_limits(
        material,
        welded=True
    )
    test_limits = np.array(test_limits['stress'][1])
    assert np.allclose(welded_values, test_limits)

    test_limits = accessories.get_pipe_stress_limits(
        material,
        welded=False
    )
    test_limits = np.array(test_limits['stress'][1])
    assert np.allclose(seamless_values, test_limits)

    with pytest.raises(KeyError,
                       match='material not found'):
        accessories.get_pipe_stress_limits('unobtainium')
