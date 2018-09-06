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


from math import sqrt
import pytest
import pint
import os
import cantera as ct
import numpy as np
import pandas as pd
from mock import patch
from .. import tube
from .. import thermochem

ureg = pint.UnitRegistry()
quant = ureg.Quantity


def compare(manual, tested):
    for key, value in manual.items():
        try:
            test_value = tested[key].to(value.units.format_babel()).magnitude
            value = value.magnitude
        except AttributeError:
            test_value = tested[key]

        assert abs(test_value - value) / value < 1e-4


class TestBolt:
    thread_size = '1/4-28'
    thread_class = '2'
    plate_max_tensile = quant(30, 'ksi')
    engagement_length = quant(0.5, 'in')
    bolt_max_tensile = quant(80, 'ksi')

    def test_calculate_stress_areas_over_100ksi(self):
        # test bolt > 100ksi
        bolt_max_tensile = quant(120, 'ksi')
        hand_calc = {
            'screw area': quant(0.034934049, 'in^2'),
            'plate area': quant(0.308744082, 'in^2'),
            'minimum engagement': quant(0.452595544, 'in')
        }
        test_areas = tube.Bolt.calculate_stress_areas(
            self.thread_size,
            self.thread_class,
            bolt_max_tensile,
            self.plate_max_tensile,
            self.engagement_length,
            ureg
        )
        compare(hand_calc, test_areas)

    def test_calculate_stress_areas_under_100ksi(self):
        # test bolt < 100ksi
        bolt_max_tensile = quant(80, 'ksi')
        hand_calc = {
            'screw area': quant(0.036374073, 'in^2'),
            'minimum engagement': quant(0.314168053, 'in')
        }
        test_areas = tube.Bolt.calculate_stress_areas(
            self.thread_size,
            self.thread_class,
            bolt_max_tensile,
            self.plate_max_tensile,
            self.engagement_length,
            ureg
        )
        compare(hand_calc, test_areas)

    def test_calculate_stress_areas_length_too_short(self):
        # ensure warning when engagement length < minimum
        engagement_length = quant(0.05, 'in')
        with pytest.warns(
                Warning,
                match='Screws fail in shear, not tension.' +
                      ' Plate may be damaged.' +
                      ' Consider increasing bolt engagement length'
        ):
            tube.Bolt.calculate_stress_areas(
                self.thread_size,
                self.thread_class,
                self.bolt_max_tensile,
                self.plate_max_tensile,
                engagement_length,
                ureg
            )

    @staticmethod
    def test_import_thread_specs():
        # good input
        test_size = '0-80'
        test_type = ['external', 'internal']
        test_property = 'pitch diameter max'
        test_classes = [['2A', '3A'], ['2B', '3B']]
        good_result = [[0.0514, 0.0519], [0.0542, 0.0536]]
        specifications = tube.Bolt._import_thread_specs()
        for thread, classes, expected in zip(test_type, test_classes,
                                             good_result):
            # check for correct output
            current_frame = specifications[thread]
            for thread_class, result in zip(classes, expected):
                test_result = current_frame[test_property][test_size][
                    thread_class]
                assert abs(test_result - result) < 1e-7

    @staticmethod
    def test_get_thread_property():
        # good input
        good_args = [
            [
                'pitch diameter max',
                '1/4-20',
                '2B',
                ureg
            ],
            [
                'pitch diameter max',
                '1/4-20',
                '2A',
                ureg
            ]
        ]
        good_results = [
            0.2248,
            0.2164
        ]
        for args, result in zip(good_args, good_results):
            test_result = tube.Bolt.get_thread_property(*args)

            assert abs(test_result.magnitude - result) < 1e-7

    @staticmethod
    def test_get_thread_property_not_in_dataframe():
        dataframes = tube.Bolt._import_thread_specs()
        # property not in dataframe
        bad_property = 'jello'
        bad_message = (
                'Thread property \'' +
                bad_property +
                '\' not found. Available specs: ' +
                "'" + "', '".join(dataframes['internal'].keys()) + "'"
        )
        with pytest.raises(
                KeyError,
                match=bad_message
        ):
            tube.Bolt.get_thread_property(
                bad_property,
                '1/4-20',
                '2B',
                ureg
            )

    @staticmethod
    def test_get_thread_property_invalid_class():
        bad_args = [
            [
                'pitch diameter max',
                '1/4-20',
                '2F',
                ureg
            ],
            [
                'pitch diameter max',
                '1/4-20',
                '6A',
                ureg
            ]
        ]
        for args in bad_args:
            with pytest.raises(
                ValueError,
                match='bad thread class'
            ):
                tube.Bolt.get_thread_property(*args)

    @staticmethod
    def test_get_thread_property_invalid_size():
        bad_size = '1290-33'
        bad_message = (
                'Thread size \'' +
                bad_size +
                '\' not found'
        )
        with pytest.raises(
                KeyError,
                match=bad_message
        ):
            tube.Bolt.get_thread_property(
                'pitch diameter max',
                bad_size,
                '2B',
                ureg
            )


class TestDDT:
    diameter = quant(5.76, ureg.inch)

    # use a unit diameter to match diameter-specific values from plot
    tube_diameter = quant(1, 'meter')

    # define gas mixture and relevant pint quantities
    mechanism = 'gri30.cti'
    gas = ct.Solution(mechanism)
    gas.TP = 300, 101325
    initial_temperature = quant(gas.T, 'K')
    initial_pressure = quant(gas.P, 'Pa')
    gas.set_equivalence_ratio(1, 'CH4', {'O2': 1, 'N2': 3.76})
    species_dict = gas.mole_fraction_dict()

    def test_calculate_spiral_diameter(self):
        # define a blockage ratio
        test_blockage_ratio = 0.44

        # define expected result and actual result
        expected_spiral_diameter = (
                self.diameter / 2 * (1 - sqrt(1 - test_blockage_ratio))
        )
        result = tube.DDT.calculate_spiral_diameter(
            self.diameter,
            test_blockage_ratio
        )

        # ensure good output
        assert expected_spiral_diameter == result.to(ureg.inch)

    def test_calculate_spiral_diameter_non_numeric_br(self):
        # ensure proper handling with non-numeric blockage ratio
        with pytest.raises(
                ValueError,
                match='Non-numeric blockage ratio.'
        ):
            tube.DDT.calculate_spiral_diameter(
                self.diameter,
                'doompity doo'
            )

    def test_calculate_spiral_diameter_bad_br(self):
        # ensure proper handling with blockage ratio outside allowable limits
        bad_blockage_ratios = [-35.124, 0, 1, 120.34]
        for ratio in bad_blockage_ratios:
            with pytest.raises(
                    ValueError,
                    match='Blockage ratio outside of 0<BR<1'
            ):
                tube.DDT.calculate_spiral_diameter(
                    self.diameter,
                    ratio
                )

    @staticmethod
    def test_calculate_blockage_ratio():
        hand_calc_blockage_ratio = 0.444242326717679

        # check for expected result with good input
        test_result = tube.DDT.calculate_blockage_ratio(
            quant(3.438, ureg.inch),
            quant(7. / 16., ureg.inch)
        )
        assert (test_result - hand_calc_blockage_ratio) < 1e-8

    @staticmethod
    def test_calculate_blockage_ratio_0_tube_diameter():
        with pytest.raises(
            ValueError,
            match='tube ID cannot be 0'
        ):
            tube.DDT.calculate_blockage_ratio(
                quant(0, ureg.inch),
                quant(7. / 16., ureg.inch)
            )

    def test_calculate_blockage_ratio_blockage_gt_tube(self):
        # check for correct handling when blockage diameter >= tube diameter
        with pytest.raises(
                ValueError,
                match='blockage diameter >= tube diameter'
        ):
            tube.DDT.calculate_blockage_ratio(
                quant(1, ureg.inch),
                quant(3, ureg.inch)
            )

    def test_calculate_ddt_run_up_bad_blockage_ratio(self):
        bad_blockages = [-4., 0, 1]
        for blockage_ratio in bad_blockages:
            with pytest.raises(
                    ValueError,
                    match='Blockage ratio outside of correlation range'
            ):
                tube.DDT.calculate_run_up(
                    blockage_ratio,
                    self.tube_diameter,
                    self.initial_temperature,
                    self.initial_pressure,
                    self.species_dict,
                    self.mechanism,
                    ureg
                )

    def test_calculate_ddt_run_up(self):
        # define good blockage ratios and expected result from each
        good_blockages = [0.1, 0.2, 0.3, 0.75]
        good_results = [
            48.51385390428211,
            29.24433249370277,
            18.136020151133494,
            4.76070528967254
        ]

        # test with good inputs
        for blockage_ratio, result in zip(good_blockages, good_results):
            test_runup = tube.DDT.calculate_run_up(
                blockage_ratio,
                self.tube_diameter,
                self.initial_temperature,
                self.initial_pressure,
                self.species_dict,
                self.mechanism,
                ureg,
                phase_specification='gri30_mix'
            )

            assert 0.5 * result <= test_runup.magnitude <= 1.5 * result


class TestWindow:
    width = quant(50, ureg.mm).to(ureg.inch)
    length = quant(20, ureg.mm).to(ureg.mile)
    pressure = quant(1, ureg.atm).to(ureg.torr)
    thickness = quant(1.2, ureg.mm).to(ureg.furlong)
    rupture_modulus = quant(5300, ureg.psi).to(ureg.mmHg)

    def test_safety_factor(self):
        desired_safety_factor = 4

        test_sf = tube.Window.safety_factor(
            self.length,
            self.width,
            self.thickness,
            self.pressure,
            self.rupture_modulus
        )

        assert abs(test_sf - desired_safety_factor) / test_sf < 0.01

    def test_minimum_thickness_sf_less_than_1(self):
        # safety factor < 1
        safety_factor = [0.25, -7]
        for factor in safety_factor:
            with pytest.raises(
                    ValueError,
                    match='Window safety factor < 1'
            ):
                tube.Window.minimum_thickness(
                    self.length,
                    self.width,
                    factor,
                    self.pressure,
                    self.rupture_modulus,
                    ureg
                )

    def test_minimum_thickness_sf_non_numeric(self):
        safety_factor = 'BruceCampbell'
        with pytest.raises(
                TypeError,
                match='Non-numeric window safety factor'
        ):
            tube.Window.minimum_thickness(
                self.length,
                self.width,
                safety_factor,
                self.pressure,
                self.rupture_modulus,
                ureg
            )

    def test_minimum_thickness(self):
        safety_factor = 4
        test_thickness = tube.Window.minimum_thickness(
            self.length,
            self.width,
            safety_factor,
            self.pressure,
            self.rupture_modulus,
            ureg
        )
        test_thickness = test_thickness.to(self.thickness.units).magnitude
        desired_thickness = self.thickness.magnitude
        assert abs(test_thickness - desired_thickness) / test_thickness < 0.01

    @staticmethod
    def test_solver_wrong_number_args():
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
            with pytest.raises(
                    ValueError,
                    match='Incorrect number of arguments sent to solver'
            ):
                tube.Window.solver(**argument_dict)

    @staticmethod
    def test_solver_bad_kwarg():
        kwargs = {
            'length': 5,
            'width': 4,
            'thickness': 2,
            'pressure': 2,
            'pulled_pork': 9
        }
        with pytest.raises(
                ValueError,
                match='Bad keyword argument:\npulled_pork'
        ):
            tube.Window.solver(**kwargs)

    @staticmethod
    def test_solver_imaginary_result():
        kwargs = {
            'length': 20,
            'width': 50,
            'safety_factor': 4,
            'pressure': -139,
            'rupture_modulus': 5300
        }
        with pytest.warns(
                Warning,
                match='Window inputs resulted in imaginary solution.'
        ):
            test_nan = tube.Window.solver(**kwargs)

        assert test_nan is np.nan

    @staticmethod
    def test_solver():
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
            test_output = tube.Window.solver(**args[index])
            assert abs(test_output - good_solutions[index]) < 0.1

    @staticmethod
    def test_calculate_window_bolt_sf():
        max_pressure = quant(1631.7, 'psi')
        window_area = quant(5.75 * 2.5, 'in^2')
        num_bolts = 20
        thread_size = '1/4-28'
        thread_class = '2'
        bolt_max_tensile = quant(120, 'ksi')
        plate_max_tensile = quant(30, 'ksi')
        engagement_length = quant(0.5, 'in')

        hand_calc = {
            'bolt': 3.606968028,
            'plate': 7.969517321,
        }

        test_values = tube.Window.calculate_bolt_sfs(
            max_pressure,
            window_area,
            num_bolts,
            thread_size,
            thread_class,
            bolt_max_tensile,
            plate_max_tensile,
            engagement_length,
            ureg
        )

        compare(hand_calc, test_values)


class TestTube:
    material = '316L'
    schedule = '80'
    nominal_size = '6'
    welded = False
    safety_factor = 4

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

    @staticmethod
    def fake_get_material_groups(*_):
        """
        fake get_material_groups()
        """
        return {'thing0': 'group0', 'thing1': 'group1'}

    def test_get_dimensions(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # Note: _get_dimensions is called during __init__
        outer_diameter = test_tube.dimensions.outer_diameter
        inner_diameter = test_tube.dimensions.inner_diameter
        wall_thickness = test_tube.dimensions.wall_thickness

        assert outer_diameter.units.format_babel() == 'inch'
        assert inner_diameter.units.format_babel() == 'inch'
        assert wall_thickness.units.format_babel() == 'inch'

        assert outer_diameter.magnitude - 6.625 < 1e-7
        assert inner_diameter.magnitude - 5.761 < 1e-7
        assert wall_thickness.magnitude - 0.432 < 1e-7

    def test_get_dimensions_bad_pipe_schedule(self):
        with pytest.raises(
                ValueError,
                match='Pipe schedule not found'
        ):
            tube.Tube(
                self.material,
                'Kropotkin',
                self.nominal_size,
                self.welded,
                self.safety_factor
            )

    def test_get_dimensions_bad_pipe_size(self):
        with pytest.raises(
                ValueError,
                match='Nominal size not found for given pipe schedule'
        ):
            tube.Tube(
                self.material,
                self.schedule,
                'really big',
                self.welded,
                self.safety_factor
            )

    def test_check_materials_list(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )
        assert test_tube._check_materials_list()

    def test_check_materials_list_no_files_found(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_listdir(*_):
            # returns an empty list to test for files-not-found condition
            return []

        with pytest.raises(
            FileNotFoundError,
            match='no files containing "flange" or "stress" found'
        ):
            with patch('os.listdir', new=fake_listdir):
                test_tube._check_materials_list()

    def test_check_materials_list_group_lookup_fails(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_listdir(*_):
            # fails group1 lookup
            return ['asdfflangegroup0sfh',
                    'asdfstressweldedasdg']

        with patch('builtins.open', new=self.FakeOpen):
            with patch('os.listdir', new=fake_listdir):
                with pytest.raises(
                        ValueError,
                        message='\nmaterial group group1 not found'
                ):
                    test_tube._check_materials_list()

    def test_check_materials_list_no_welded_or_seamless(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_listdir(*_):
            # causes welded/seamless warning
            return ['asdfflangegroup0sfh',
                    'asdfflangegroup1asd',
                    'asdfstresswdasdg']

        with patch('builtins.open', new=self.FakeOpen):
            with patch(
                    'beaverdet.tube.Tube._get_material_groups',
                    self.fake_get_material_groups
            ):
                with patch('os.listdir', new=fake_listdir):
                    error_string = 'asdfstresswdasdg' + \
                                   'does not indicate whether it is welded' + \
                                   ' or seamless'
                    with pytest.warns(Warning, match=error_string):
                        test_tube._check_materials_list()

    def test_check_materials_list_missing_material(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            '..',
            'lookup_data'
        )

        def fake_listdir(*_):
            """
            listdir function that should work 100%
            """
            return ['asdfflangegroup0sfh',
                    'asdfflangegroup1asd',
                    'asdfstressweldedasdg']

        with patch('builtins.open', new=self.FakeOpen):
            with patch(
                    'beaverdet.tube.Tube._get_material_groups',
                    self.fake_get_material_groups
            ):
                with patch('os.listdir', new=fake_listdir):
                    class NewFakeFile:
                        """
                        FakeFile class that should fail material lookup
                        """

                        @staticmethod
                        def readline(*_):
                            """
                            readline function that should fail material
                            lookup for thing1
                            """
                            return 'ASDF,thing0\n'

                    setattr(self.FakeOpen, 'FakeFile', NewFakeFile)
                    error_string = '\nMaterial thing1 not found in ' + \
                                   os.path.join(
                                       file_directory,
                                       'asdfstressweldedasdg'
                                   )
                    with pytest.raises(ValueError, message=error_string):
                                test_tube._check_materials_list()

    def test_get_material_groups(self):
        # ensure correctness by comparing to a pandas dataframe reading the
        # same file
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # file information
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            '..',
            'lookup_data'
        )
        file_name = 'materials_list.csv'
        file_location = os.path.join(file_directory, file_name)

        # load data into a test dataframe
        test_dataframe = pd.read_csv(file_location)

        # load data into a dictionary using get_material_groups()
        test_output = test_tube._get_material_groups()

        # collect keys and values from dataframe that should correspond to
        # those of the dictionary
        keys_from_dataframe = test_dataframe.Grade.values.astype(str)
        values_from_dataframe = test_dataframe.Group.values.astype(str)

        for index, key in enumerate(keys_from_dataframe):
            # make sure each set of values are approximately equal
            dict_value = test_output[key]
            dataframe_value = values_from_dataframe[index]
            assert dict_value == dataframe_value

    def test_get_material_groups_nonexistent_file(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_exists(*_):
            return False

        with patch(
            'os.path.exists',
            new=fake_exists
        ):
            with pytest.raises(
                    ValueError,
                    match='materials_list.csv does not exist'
            ):
                test_tube._get_material_groups()

    def test_get_material_groups_blank_file(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_csv_import(*_):
            raise pd.errors.EmptyDataError

        with patch(
            'pandas.read_csv',
            new=fake_csv_import
        ):
            # check for proper error handling when file is blank
            with pytest.raises(
                    ValueError,
                    match='materials_list.csv is empty'
            ):
                test_tube._get_material_groups()

    def test_collect_materials(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )
        columns = ['Grade', 'Group', 'ElasticModulus', 'Density', 'Poisson']
        data = [
            ['304', 2.1, 200, 7.8, 0.28],
            ['304H', 2.1, 200, 7.8, 0.28],
            ['316', 2.2, 200, 7.9, 0.28],
            ['316H', 2.2, 200, 7.9, 0.28],
            ['317', 2.2, 200, 7.9, 0.28],
            ['304L', 2.3, 200, 7.8, 0.28],
            ['316L', 2.3, 200, 7.9, 0.28]
        ]
        good_dataframe = pd.DataFrame(
            data=data,
            index=None,
            columns=columns
        )
        good_dataframe['ElasticModulus'] = [
            test_tube._units.quant(item, 'GPa') for item in
            good_dataframe['ElasticModulus']
        ]
        good_dataframe['Density'] = [
            test_tube._units.quant(item, 'g/cm^3') for item in
            good_dataframe['Density']
        ]

        test_dataframe = test_tube._collect_tube_materials()

        assert test_dataframe.equals(good_dataframe)

    def test_collect_materials_nonexistent_file(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_exists(*_):
            return False

        with patch(
            'os.path.exists',
            new=fake_exists
        ):
            with pytest.raises(
                    ValueError,
                    match='materials_list.csv does not exist'
            ):
                test_tube._get_material_groups()

    def test_collect_materials_empty_file(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        def fake_csv_import(*_):
            raise pd.errors.EmptyDataError

        with patch(
            'pandas.read_csv',
            new=fake_csv_import
        ):
            # check for proper error handling when file is blank
            with pytest.raises(
                    ValueError,
                    match='materials_list.csv is empty'
            ):
                test_tube._get_material_groups()

    def test_get_flange_limits_from_csv(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # file information
        my_input = 'testfile'
        file_directory = os.path.join(
            os.path.dirname(os.path.relpath(__file__)),
            '..', 'lookup_data')
        file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
        file_location = os.path.relpath(os.path.join(file_directory, file_name))

        # create test dataframe and write it to a .csv file
        good_dataframe = pd.DataFrame(data=[[0, 1],  # temperatures
                                            [2, 3]],  # pressures
                                      columns=['Temperature', 'Class'])
        good_dataframe.to_csv(file_location, index=False)

        # add units to test dataframe
        good_dataframe['Temperature'] = [
            test_tube._units.quant(temp, 'degC') for temp in
            good_dataframe['Temperature']
        ]
        good_dataframe['Class'] = [
            test_tube._units.quant(pressure, 'bar') for pressure in
            good_dataframe['Class']
        ]

        # read in test dataframe using get_flange_limits_from_csv()
        test_dataframe = test_tube._get_flange_limits_from_csv(my_input)

        os.remove(file_location)

        assert test_dataframe.equals(good_dataframe)

    def test_get_flange_limits_from_csv_bad_group(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        my_input = 'batman'
        bad_output = my_input + ' is not a valid group'

        # ensure the error is handled properly
        with pytest.raises(
                ValueError,
                match=bad_output
        ):
            test_tube._get_flange_limits_from_csv(my_input)

    def test_get_flange_limits_from_csv_bad_pressure(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # file information
        my_input = 'testfile'
        file_directory = os.path.join(
            os.path.dirname(os.path.relpath(__file__)),
            '..', 'lookup_data')
        file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
        file_location = os.path.relpath(os.path.join(file_directory, file_name))

        # create test dataframe and write it to a .csv file
        bad_dataframe = pd.DataFrame(data=[[0, 1],  # temperatures
                                            [2, -3]],  # pressures
                                     columns=['Temperature', 'Class'])

        bad_dataframe.to_csv(file_location, index=False)

        with pytest.raises(
                ValueError,
                match='Pressure less than zero.'
        ):
            test_tube._get_flange_limits_from_csv(my_input)

        os.remove(file_location)

    def test_get_flange_limits_from_csv_zeroed_non_numeric(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # file information
        my_input = 'testfile'
        file_directory = os.path.join(
            os.path.dirname(os.path.relpath(__file__)),
            '..', 'lookup_data')
        file_name = 'ASME_B16_5_flange_ratings_group_' + my_input + '.csv'
        file_location = os.path.relpath(os.path.join(file_directory, file_name))

        # create test dataframe and write it to a .csv file
        test_temperatures = [9, 's', 'd']
        good_temperatures = [9, 0, 0]
        test_pressures = ['a', 3, 'f']
        good_pressures = [0, 3, 0]
        test_dataframe = pd.DataFrame({'Temperature': test_temperatures,
                                       'Pressure': test_pressures})
        good_dataframe = pd.DataFrame({'Temperature': good_temperatures,
                                       'Pressure': good_pressures})

        # add units to test dataframe
        good_dataframe['Temperature'] = [
            test_tube._units.quant(temp, 'degC') for temp in
            good_dataframe['Temperature']
        ]
        good_dataframe['Pressure'] = [
            test_tube._units.quant(pressure, 'bar') for pressure in
            good_dataframe['Pressure']
        ]

        test_dataframe.to_csv(file_location, index=False)

        # ensure non-numeric pressures and temperatures are zeroed out
        test_dataframe = test_tube._get_flange_limits_from_csv(my_input)

        assert test_dataframe.equals(good_dataframe)

    def test_get_pipe_stress_limits_welded(self):
        test_tube = tube.Tube(
            '304',
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # known values for 304
        welded_values = np.array(
            [16, 16, 13.3, 12, 11, 10.5, 9.7, 9.5, 9.4, 9.2, 9,
             8.8, 8.7, 8.5, 8.3, 8.1, 7.6, 6.5, 5.4])

        test_limits = test_tube._get_pipe_stress_limits(
            welded=True
        )
        test_limits = np.array(test_limits['stress'][1])
        assert np.allclose(welded_values, test_limits)

    def test_get_pipe_stress_limits_seamless(self):
        test_tube = tube.Tube(
            '304',
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # known values for 304
        seamless_values = np.array(
            [18.8, 18.8, 15.7, 14.1, 13, 12.2, 11.4, 11.3,
             11.1, 10.8, 10.6, 10.4, 10.2, 10, 9.8, 9.5, 8.9,
             7.7, 6.1])

        test_limits = test_tube._get_pipe_stress_limits(
            welded=False
        )
        test_limits = np.array(test_limits['stress'][1])
        assert np.allclose(seamless_values, test_limits)

    def test_get_pipe_dlf(self):
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
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        # from hand calcs, critical velocity is 1457.44 m/s, giving upper and
        # lower bounds of 1603.188 and 1311.700 m/s
        cj_speeds = [
            test_tube._units.quant(1200, 'm/s'),  # DLF 1
            test_tube._units.quant(1311, 'm/s'),  # DLF 1
            test_tube._units.quant(1312, 'm/s'),  # DLF 4
            test_tube._units.quant(1400, 'm/s'),  # DLF 4
            test_tube._units.quant(1603, 'm/s'),  # DLF 4
            test_tube._units.quant(1604, 'm/s'),  # DLF 2
            test_tube._units.quant(2000, 'm/s')  # DLF 2
        ]
        expected_dlf = [1, 1, 4, 4, 4, 2, 2]
        test_dlf = []
        for cj_speed in cj_speeds:
            test_tube.cj_speed = cj_speed
            test_dlf.append(
                test_tube._get_pipe_dlf(plus_or_minus=0.1)
            )

        assert all(
            dlf == good_dlf for dlf, good_dlf in zip(test_dlf, expected_dlf)
        )

    def test_get_pipe_dlf_bad_plus_minus_value(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        test_tube.cj_speed = test_tube._units.quant(1200, 'm/s')

        # plus_or_minus outside of (0, 1)
        bad_plus_minus = [-1, 0, 1, 2]
        for plus_minus in bad_plus_minus:
            with pytest.raises(
                ValueError,
                match='plus_or_minus factor not between 0 and 1'
            ):
                test_tube._get_pipe_dlf(plus_minus)

    def test_calculate_max_stress_seamless(self):
        test_tube = tube.Tube(
            '316L',
            self.schedule,
            self.nominal_size,
            False,
            self.safety_factor
        )

        initial_temperatures = [
            test_tube._units.quant(100, 'degF'),
            test_tube._units.quant(200, 'degF'),
            test_tube._units.quant(150, 'degF')
        ]

        good_stresses = [
            test_tube._units.quant(15.7, 'ksi'),
            test_tube._units.quant(13.3, 'ksi'),
            test_tube._units.quant(14.5, 'ksi')
        ]
        for temperature, stress in zip(initial_temperatures, good_stresses):
            assert np.allclose(
                test_tube.calculate_max_stress(temperature),
                stress
            )

    def test_calculate_max_stress_welded(self):
        test_tube = tube.Tube(
            '304',
            self.schedule,
            self.nominal_size,
            True,
            self.safety_factor
        )

        initial_temperatures = [
            test_tube._units.quant(650, 'degF'),
            test_tube._units.quant(700, 'degF'),
            test_tube._units.quant(675, 'degF')
        ]

        good_stresses = [
            test_tube._units.quant(9.5, 'ksi'),
            test_tube._units.quant(9.4, 'ksi'),
            test_tube._units.quant(9.45, 'ksi')
        ]
        for temperature, stress in zip(initial_temperatures, good_stresses):
            assert np.allclose(
                test_tube.calculate_max_stress(temperature),
                stress
            )

    def test_calculate_max_stress_non_monotonic(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        temp = test_tube._units.quant(100, 'degF')

        def make_non_monotonic(*_):
            return {
                'temperature':('degF', [0, 1, -1]),
                'stress':('ksi', [0, 0, 0])
            }

        with patch(
            'beaverdet.tube.Tube._get_pipe_stress_limits',
            new=make_non_monotonic
        ):
            with pytest.raises(
                ValueError,
                match='Stress limits require temperatures to be ' +
                      'monotonically increasing'
            ):
                test_tube.calculate_max_stress(temp)

    def test_calculate_max_pressure(self):
        # TODO: write this test
        assert False

    def test_calculate_initial_pressure(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        initial_temperature = test_tube._units.quant(300, 'K')
        test_tube.calculate_max_stress(initial_temperature)

        species_dict = {'H2': 1, 'O2': 0.5}
        mechanism = 'gri30.cti'
        max_pressures = [
            test_tube.calculate_max_pressure(
                test_tube._units.quant(1200, 'psi')
            ),
            test_tube.calculate_max_pressure()
        ]
        error_tol = 1e-4

        max_solutions = [max_pressures[0], quant(149.046409603932, 'atm')]

        # test function output
        for max_pressure, max_solution in zip(max_pressures, max_solutions):
            test_tube.max_pressure = max_pressure
            test_result = test_tube.calculate_initial_pressure(
                species_dict,
                mechanism,
                error_tol=error_tol
            )

            states = thermochem.calculate_reflected_shock_state(
                test_tube.initial_temperature,
                test_result,
                species_dict,
                mechanism,
                test_tube._units.ureg
            )

            # get dynamic load factor
            test_tube.cj_speed = states['cj']['speed']
            dlf = test_tube._get_pipe_dlf()

            calc_max = states['reflected']['state'].P
            max_solution = max_solution.to('Pa').magnitude / dlf

            error = abs(max_solution - calc_max) / max_solution

            assert error <= 0.0005

    def test_lookup_flange_class(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )
        test_tube.max_pressure = test_tube._units.quant(125, 'bar')
        test_tube.initial_temperature = test_tube._units.quant(350, 'degC')
        assert test_tube.lookup_flange_class() == '1500'

    def test_lookup_flange_class_bad_temperature(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        test_temperatures = [
            test_tube._units.quant(-100, 'degC'),
            test_tube._units.quant(500, 'degC')
        ]

        test_tube.max_pressure = test_tube._units.quant(125, 'bar')

        for temperature in test_temperatures:
            test_tube.initial_temperature = temperature
            with pytest.raises(
                    ValueError,
                    match='Temperature out of range.'
            ):
                test_tube.lookup_flange_class()

    def test_lookup_flange_class_bad_pressure(self):
        test_tube = tube.Tube(
            self.material,
            self.schedule,
            self.nominal_size,
            self.welded,
            self.safety_factor
        )

        test_tube.initial_temperature = test_tube._units.quant(350, 'degC')

        test_pressures = [
            test_tube._units.quant(-10, ureg.bar),
            test_tube._units.quant(350, ureg.bar)
        ]

        for pressure in test_pressures:
            test_tube.max_pressure = pressure
            with pytest.raises(
                    ValueError,
                    match='Pressure out of range.'
            ):
                test_tube.lookup_flange_class()
