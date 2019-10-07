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


# noinspection PyProtectedMember
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
            # noinspection PyTypeChecker
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
            # noinspection PyTypeChecker
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


# noinspection PyProtectedMember
class TestTube:
    material = '316L'
    schedule = '80'
    nominal_size = '6'
    welded = False
    safety_factor = 4
    fake_material_groups = {'thing0': 'group0', 'thing1': 'group1'}
    fake_materials = pd.DataFrame(columns=['Group', 'Grade'])

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

    def test__dimensions_lookup_bad_schedule(self):
        test_tube = tube.Tube()
        test_tube._set_property('schedule', 'garbage')

        with pytest.raises(
            ValueError,
            match='\nPipe schedule not found'
        ):
            test_tube._dimensions_lookup()

    def test__dimensions_lookup_bad_size(self):
        test_tube = tube.Tube()
        test_tube._set_property('nominal_size', '2.222')

        with pytest.raises(
            ValueError,
            match='\nNominal size not found for given pipe schedule'
        ):
            test_tube._dimensions_lookup()

    def test_prop_autocalc_initial_set_get(self):
        inputs = [True, False, 0, 1]
        expected = [bool(item) for item in inputs]
        test_tube = tube.Tube()
        results = []
        for value, correct in zip(inputs, expected):
            test_tube.autocalc_initial = value
            results.append(value == correct)

        assert all(results)

    def test_prop_show_warnings_set_get(self):
        inputs = [True, False, 0, 1]
        expected = [bool(item) for item in inputs]
        test_tube = tube.Tube()
        results = []
        for value, correct in zip(inputs, expected):
            test_tube.show_warnings = value
            results.append(value == correct)

        assert all(results)

    def test_prop_available_pipe_sizes_set(self):
        test_tube = tube.Tube()

        with pytest.raises(
            PermissionError,
            match='\nPipe sizes can not be set manually.'
        ):
            test_tube.available_pipe_sizes = 7

    def test_prop_available_pipe_sizes_get(self):
        test_tube = tube.Tube()
        assert isinstance(test_tube.available_pipe_sizes, list)

    def test_prop_available_pipe_schedules_set(self):
        test_tube = tube.Tube()

        with pytest.raises(
            PermissionError,
            match='\nPipe schedules can not be set manually.'
        ):
            test_tube.available_pipe_schedules = 'big as hell'

    def test_prop_available_pipe_schedules_get(self):
        test_tube = tube.Tube()
        assert isinstance(test_tube.available_pipe_schedules, list)

    def test_prop_available_tube_materials_set(self):
        test_tube = tube.Tube()

        with pytest.raises(
            PermissionError,
            match='\nAvailable tube materials can not be set manually.'
        ):
            test_tube.available_tube_materials = 'Aluminum Foil'

    def test_prop_available_tube_materials_get(self):
        test_tube = tube.Tube()
        assert isinstance(test_tube.available_tube_materials, list)

    def test_prop_nominal_size_set_get(self):
        test_tube = tube.Tube()
        inputs = [1, 6, '1 1/2']
        expected = [str(item) for item in inputs]
        results = []
        for value, correct in zip(inputs, expected):
            test_tube.nominal_size = value
            result = test_tube.nominal_size
            results.append(result == correct)

        assert all(results)

    def test_prop_nominal_size_set_bad_size(self):
        bad_size = 'really big'
        match_string = '\n{0} is not a valid pipe size. '.format(bad_size) + \
                       'For a list of available sizes, try \n' + \
                       '`mytube.available_pipe_sizes`'
        with pytest.raises(
                ValueError,
                match=match_string
        ):
            test_tube = tube.Tube()
            test_tube.nominal_size = bad_size

    def test_prop_schedule_set_get(self):
        test_tube = tube.Tube()
        inputs = [40, 'XXH']
        expected = [str(item) for item in inputs]
        results = []
        for value, correct in zip(inputs, expected):
            test_tube.schedule = value
            result = test_tube.schedule
            results.append(result == correct)

        assert all(results)

    def test_prop_schedule_set_bad_schedule(self):
        bad_schedule = 'Kropotkin'
        match_string = '\n{0} is not a valid pipe schedule for this nominal' \
                       ' size. '.format(bad_schedule) + \
                       'For a list of available schedules, try \n' + \
                       '`mytube.available_pipe_schedules`'
        with pytest.raises(
                ValueError,
                match=match_string
        ):
            test_tube = tube.Tube()
            test_tube.schedule = bad_schedule

    def test_prop_dimensions_set(self):
        test_tube = tube.Tube()
        match_string = '\nTube dimensions are looked up based on nominal' \
                       ' pipe size and schedule, not set. Try ' \
                       '`mytube.schedule()` or `mytube.nominal_size()` instead.'

        with pytest.raises(PermissionError) as err:
            test_tube.dimensions = [0, 2, 3]

        assert str(err.value) == match_string

    def test_prop_dimensions_get(self):
        test_tube = tube.Tube(
            nominal_size=self.nominal_size,
            schedule=self.schedule
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

    def test_prop_material_set_bad_material(self):
        test_tube = tube.Tube()
        match_string = '\nPipe material not found. For a list of ' + \
                       'available materials try:\n' + \
                       '`mytube.available_tube_materials`'
        with pytest.raises(
            ValueError,
            match=match_string
        ):
            test_tube.material = 'unobtainium'

    def test_prop_welded(self):
        welded_args = [0, 1, True, False]
        welded_results = [False, True, True, False]
        test_tube = tube.Tube()

        tests = []
        for welded_in, good_result in zip(welded_args, welded_results):
            test_tube.welded = welded_in
            tests.append(test_tube.welded == good_result)

        assert all(tests)

    def test_check_materials_list(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )
        assert test_tube._check_materials_list()

    def test_check_materials_list_no_files_found(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
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
        test_tube = tube.Tube()

        # there is no group0 in any of the flange rating groups, and trying to
        # look it up should generate a material group not found portion of the
        # error string. This test sets the materials dataframe to an empty one
        # with only Group and Grade columns, then adds a material group
        # 'group0' to generate the desired error. Grade is set to 316L to
        # avoid a grade-related error.
        try:
            test_tube._materials = self.fake_materials.copy()
            test_tube._materials['Group'] = ['group0']
            test_tube._materials['Grade'] = ['316L']
            test_tube._check_materials_list()
        except ValueError as err:
            message = '\nmaterial group group0 not found\n'
            assert message == str(err)

    def test_check_materials_list_no_welded_or_seamless(self):
        test_tube = tube.Tube()

        def fake_listdir(*_):
            # causes welded/seamless warning
            return ['asdfflangegroup0sfh',
                    'asdfflangegroup1asd',
                    'asdfstresswdasdg']

        test_tube._material_groups = self.fake_material_groups
        test_tube._materials = self.fake_materials

        with patch('builtins.open', new=self.FakeOpen):
            with patch('os.listdir', new=fake_listdir):
                error_string = 'asdfstresswdasdg' + \
                               'does not indicate whether it is welded' + \
                               ' or seamless'
                with pytest.warns(UserWarning, match=error_string):
                    test_tube._check_materials_list()

    def test_check_materials_list_missing_material(self):
        test_tube = tube.Tube()

        file_directory = os.path.join(
            'beaverdet',
            'lookup_data'
        )

        def fake_listdir(*_):
            """
            listdir function that should work 100%
            """
            return ['asdfflangegroup0sfh',
                    'asdfflangegroup1asd',
                    'asdfstressweldedasdg']

        test_tube._material_groups = self.fake_material_groups
        test_tube._materials = pd.DataFrame(
            columns=['Grade', 'Group'],
            data=list(map(list, zip(
                self.fake_material_groups.keys(),
                self.fake_material_groups.values()
            )))
        )

        with patch('builtins.open', new=self.FakeOpen):
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
                               ) + '\n'
                try:
                    test_tube._check_materials_list()
                except ValueError as err:
                    assert str(err) == error_string

    def test_get_material_groups(self):
        # ensure correctness by comparing to a pandas dataframe reading the
        # same file
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
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
        test_output = test_tube._material_groups

        # collect keys and values from dataframe that should correspond to
        # those of the dictionary
        keys_from_dataframe = test_dataframe.Grade.values.astype(str)
        values_from_dataframe = test_dataframe.Group.values.astype(str)

        for index, key in enumerate(keys_from_dataframe):
            # make sure each set of values are approximately equal
            dict_value = test_output[key]
            dataframe_value = values_from_dataframe[index]
            assert dict_value == dataframe_value

    def test_collect_tube_materials(self):
        test_tube = tube.Tube()
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

        assert test_tube._materials.equals(good_dataframe)

    def test_collect_tube_materials_nonexistent_file(self):
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
                tube.Tube()

    def test_collect_tube_materials_empty_file(self):
        # noinspection PyUnresolvedReferences
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
                tube.Tube()

    def test_get_flange_limits_from_csv(self):
        test_tube = tube.Tube()
        flange_class = '900'
        temperature = test_tube._units.quant(400, 'degC')

        # good pressures manually entered based on values from .csv files
        good_pressures = {
            '2.1': 85.3,
            '2.2': 88.3,
            '2.3': 72.9
        }

        tests = []
        for group, pressure in good_pressures.items():
            # read the imported pressure and compare it to the correct pressure
            test_pressure = test_tube._flange_limits[group] \
                .loc[test_tube._flange_limits[group]['Temperature'] ==
                     temperature][flange_class].values[0]
            good_pressure = test_tube._units.quant(pressure, 'bar')
            tests.append(test_pressure == good_pressure)

        assert all(tests)

    def test_get_flange_limits_from_csv_bad_group(self):
        test_tube = tube.Tube()
        file_name = 'ASME_B16_5_flange_ratings_group_2_1.csv'
        file_directory = os.path.join(
                os.path.dirname(os.path.relpath(tube.__file__)),
                'lookup_data')
        file_location = os.path.join(
            file_directory,
            file_name
        )
        match_string = '\n' + file_location + 'not found'

        # bad group is identified by checking the existence of a .csv fil
        def fake_os_path_exists(*_):
            return False

        with patch(
            'os.path.exists',
            new=fake_os_path_exists
        ):
            try:
                test_tube._get_flange_limits_from_csv()
            except FileNotFoundError as err:
                err = str(err)
                assert match_string == err

    def test_get_flange_limits_from_csv_bad_pressure(self):
        test_tube = tube.Tube()
        match_string = '\nPressure less than zero.'

        # an error should be thrown when a non-temperature column has a
        # negative value
        def fake_read_csv(*_):
            return pd.DataFrame(columns=['Temperature', '900'],
                                data=[[10, -10]])

        with patch(
                'pandas.read_csv',
                new=fake_read_csv
        ):
            with pytest.raises(
                ValueError,
                match=match_string
            ):
                test_tube._get_flange_limits_from_csv()

    def test_get_flange_limits_from_csv_zeroed_non_numeric(self):
        test_tube = tube.Tube()

        # create test dataframe and a dataframe of what the output should
        # look like after the test dataframe is imported
        test_temperatures = [9, 's', 'd']
        good_temperatures = [9., 0., 0.]
        test_pressures = ['a', 3, 'f']
        good_pressures = [0., 3., 0.]
        test_dataframe = pd.DataFrame({'Temperature': test_temperatures,
                                       'Pressure': test_pressures})
        good_dataframe = pd.DataFrame({'Temperature': good_temperatures,
                                       'Pressure': good_pressures})

        # convince the tube to load the test dataframe without having to
        # write a .csv to disk
        def fake_read_csv(*_):
            return test_dataframe

        with patch(
                'pandas.read_csv',
                new=fake_read_csv
        ):
            test_tube._get_flange_limits_from_csv()

            tests = []
            for group in set(test_tube._material_groups.values()):
                # pull the dataframe of flange limits for each material group
                # and use applymap to remove units from all pint quantities.
                # This is done to circumvent an error caused by using
                # np.allclose for an array of pint quantities. Comparison to
                # good_dataframe.values is done to avoid having to care about
                # how the values are distributed by pandas.
                df = test_tube._flange_limits[group]
                test_values = df.applymap(lambda x: x.magnitude)
                test = np.allclose(test_values,
                                   good_dataframe.values)
                tests.append(test)

        assert all(tests)

    def test_get_pipe_stress_limits_welded(self):
        test_tube = tube.Tube(
            material='304',
            welded=True
        )

        # known values for 304
        welded_values = np.array(
            [16, 16, 13.3, 12, 11, 10.5, 9.7, 9.5, 9.4, 9.2, 9,
             8.8, 8.7, 8.5, 8.3, 8.1, 7.6, 6.5, 5.4])

        test_limits = test_tube._get_pipe_stress_limits()
        test_limits = np.array(test_limits['stress'][1])
        assert np.allclose(welded_values, test_limits)

    def test_get_pipe_stress_limits_seamless(self):
        test_tube = tube.Tube(
            material='304',
            welded=False
        )

        # known values for 304
        seamless_values = np.array(
            [18.8, 18.8, 15.7, 14.1, 13, 12.2, 11.4, 11.3,
             11.1, 10.8, 10.6, 10.4, 10.2, 10, 9.8, 9.5, 8.9,
             7.7, 6.1])

        test_limits = test_tube._get_pipe_stress_limits()
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
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
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
            test_tube._units.quant(2000, 'm/s')   # DLF 2
        ]
        expected_dlf = [1, 1, 4, 4, 4, 2, 2]
        test_dlf = []
        for cj_speed in cj_speeds:
            test_dlf.append(
                test_tube._get_pipe_dlf(plus_or_minus=0.1, cj_vel=cj_speed)
            )

        assert all(
            dlf == good_dlf for dlf, good_dlf in zip(test_dlf, expected_dlf)
        )

    def test_get_pipe_dlf_bad_plus_minus_value(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )

        # plus_or_minus outside of (0, 1)
        bad_plus_minus = [-1, 0, 1, 2]
        for plus_minus in bad_plus_minus:
            with pytest.raises(
                ValueError,
                match='plus_or_minus factor not between 0 and 1'
            ):
                test_tube._get_pipe_dlf(
                    plus_or_minus=plus_minus,
                    cj_vel=test_tube._units.quant(2, 'm/s')
                )

    def test_calculate_max_stress_seamless(self):
        test_tube = tube.Tube(
            material='316L',
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=False,
            safety_factor=self.safety_factor
        )

        initial_temperatures = [
            test_tube._units.quant(100, 'degF'),
            test_tube._units.quant(200, 'degF'),
            test_tube._units.quant(150, 'degF')
        ]

        # in ksi
        good_stresses = [
            15.7,
            13.3,
            14.5
        ]
        for temperature, stress in zip(initial_temperatures, good_stresses):
            test_tube.initial_temperature = temperature
            assert np.allclose(
                test_tube.calculate_max_stress().to('ksi').magnitude,
                stress
            )

    def test_calculate_max_stress_welded(self):
        test_tube = tube.Tube(
            material='304',
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=True,
            safety_factor=self.safety_factor
        )

        initial_temperatures = [
            test_tube._units.quant(650, 'degF'),
            test_tube._units.quant(700, 'degF'),
            test_tube._units.quant(675, 'degF')
        ]

        # in ksi
        good_stresses = [
            9.5,
            9.4,
            9.45
        ]
        for temperature, stress in zip(initial_temperatures, good_stresses):
            test_tube.initial_temperature = temperature
            assert np.allclose(
                test_tube.calculate_max_stress().to('ksi').magnitude,
                stress
            )

    def test_calculate_max_stress_non_monotonic(self):
        test_tube = tube.Tube()

        temp = test_tube._units.quant(100, 'degF')

        def make_non_monotonic(*_):
            return {
                'temperature': ('degF', [0, 1, -1]),
                'stress': ('ksi', [0, 0, 0])
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
                test_tube.initial_temperature = temp

    @staticmethod
    def test_calculate_max_pressure():
        safety_factor = 4
        test_tube = tube.Tube(
            material='304',
            schedule='80',
            nominal_size='6',
            welded=False,
            safety_factor=safety_factor
        )

        max_stress = test_tube._units.quant(18.8, 'ksi')
        wall_thickness = test_tube._units.quant(0.432, 'in')
        outer_diameter = test_tube._units.quant(6.625, 'in')
        inner_diameter = outer_diameter - 2 * wall_thickness
        mean_diameter = (inner_diameter + outer_diameter) / 2
        asme_fs = 4

        good_max_pressure = (
                max_stress * (2 * wall_thickness) * asme_fs /
                (mean_diameter * safety_factor)
                             ).to('Pa').magnitude

        test_tube.max_stress = max_stress

        test_max_pressure = test_tube.calculate_max_pressure()\
            .to('Pa').magnitude

        assert np.allclose(test_max_pressure, good_max_pressure)

    def test_calculate_initial_pressure_no_mp(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor,
            mechanism='gri30.cti',
            fuel='H2',
            oxidizer='O2',
            equivalence_ratio=1,
            show_warnings=False,
            initial_temperature=(300, 'K'),
            autocalc_initial=True,
            use_multiprocessing=False
        )
        # the initial pressure should cause the reflected detonation pressure
        # to be equal to the tube's max pressure, accounting for dynamic load
        # factor
        correct_max = test_tube.max_pressure.to('Pa').magnitude
        test_state = thermochem.calculate_reflected_shock_state(
            test_tube.initial_temperature,
            test_tube.initial_pressure,
            test_tube.reactant_mixture,
            test_tube.mechanism,
            test_tube._units.ureg
        )

        error = abs(
            correct_max -
            test_state['reflected']['state'].P * test_tube.dynamic_load_factor
        ) / correct_max

        assert error <= 0.0005

    def test_calculate_initial_pressure_with_mp(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor,
            mechanism='gri30.cti',
            fuel='H2',
            oxidizer='O2',
            equivalence_ratio=1,
            show_warnings=False,
            initial_temperature=(300, 'K'),
            autocalc_initial=True,
            use_multiprocessing=True,
            verbose=True
        )
        # the initial pressure should cause the reflected detonation pressure
        # to be equal to the tube's max pressure, accounting for dynamic load
        # factor
        correct_max = test_tube.max_pressure.to('Pa').magnitude
        test_state = thermochem.calculate_reflected_shock_state(
            test_tube.initial_temperature,
            test_tube.initial_pressure,
            test_tube.reactant_mixture,
            test_tube.mechanism,
            test_tube._units.ureg
        )

        error = abs(
            correct_max -
            test_state['reflected']['state'].P * test_tube.dynamic_load_factor
        ) / correct_max

        assert error <= 0.0005

    def test_lookup_flange_class(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )
        test_tube.max_pressure = test_tube._units.quant(125, 'bar')
        test_tube.initial_temperature = test_tube._units.quant(350, 'degC')
        assert test_tube.lookup_flange_class() == 1500

    def test_lookup_flange_class_bad_material(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )
        test_tube.max_pressure = test_tube._units.quant(125, 'bar')
        test_tube.initial_temperature = test_tube._units.quant(350, 'degC')
        assert test_tube.lookup_flange_class() == 1500

    def test_lookup_flange_class_bad_temperature(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )

        test_temperatures = [
            test_tube._units.quant(-100, 'degC'),
            test_tube._units.quant(500, 'degC')
        ]

        test_tube.max_pressure = test_tube._units.quant(125, 'bar')

        for temperature in test_temperatures:
            # fake out the initial temperature set, because using the property
            # directly causes flange class to be calculated. Direct calculation
            # isn't desirable for this test because it is less direct.
            test_tube._properties['initial_temperature'] = temperature
            with pytest.raises(
                    ValueError,
                    match='Temperature out of range.'
            ):
                test_tube.lookup_flange_class()

    def test_lookup_flange_class_bad_pressure(self):
        test_tube = tube.Tube(
            material=self.material,
            schedule=self.schedule,
            nominal_size=self.nominal_size,
            welded=self.welded,
            safety_factor=self.safety_factor
        )

        test_tube.initial_temperature = test_tube._units.quant(350, 'degC')

        test_pressure = test_tube._units.quant(350, ureg.bar)
        # fake out the initial pressure set, because using the property
        # directly causes flange class to be calculated. Direct calculation
        # isn't desirable for this test because it is less direct.
        test_tube._properties['max_pressure'] = test_pressure
        with pytest.raises(
                ValueError,
                match='Pressure out of range.'
        ):
            test_tube.lookup_flange_class()
