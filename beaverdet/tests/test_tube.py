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
import cantera as ct
import numpy as np
from .. import tube

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
    pass


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
    test_class = tube.lookup_flange_class(temp_good, press_good, material)
    assert test_class == '1500'

    # check for error handling with temperature too low/high
    test_temperatures = [temp_low, temp_high]
    for temperature in test_temperatures:
        with pytest.raises(ValueError, match='Temperature out of range.'):
            tube.lookup_flange_class(temperature, press_good, material)

    # check for error handling with pressure too low/high
    test_pressures = [press_low, press_high]
    for pressure in test_pressures:
        with pytest.raises(ValueError, match='Pressure out of range.'):
            tube.lookup_flange_class(temp_good, pressure, material)

    # check for error handling when material isn't in database
    with pytest.raises(ValueError, match='Desired material not in database.'):
        tube.lookup_flange_class(temp_good, press_good, 'unobtainium')

    # check for error handling with non-string material
    bad_materials = [0, 3.14, -7]
    for bad_material in bad_materials:
        with pytest.raises(ValueError,
                           match='Desired material non-string input.'):
            tube.lookup_flange_class(temp_good, press_good, bad_material)


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
        test_dlf = tube.get_pipe_dlf(
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
            tube.get_pipe_dlf(
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
        tube.get_pipe_dlf(
            pipe_material,
            pipe_schedule,
            nominal_pipe_size,
            cj_speed
        )


def test_calculate_max_initial_pressure():
        ureg = pint.UnitRegistry()
        quant = ureg.Quantity

        # define required variables
        pipe_material = '316L'
        pipe_schedule = '80'
        pipe_nps = '6'
        welded = False
        desired_fs = 4
        initial_temperature = quant(300, 'K')
        species_dict = {'H2': 1, 'O2': 0.5}
        mechanism = 'gri30.cti'
        max_pressures = [quant(1200, 'psi'), False]
        error_tol = 1e-4

        max_solutions = [max_pressures[0], quant(149.046409603932, 'atm')]

        # test function output
        for max_pressure, max_solution in zip(max_pressures, max_solutions):
            test_result = tube.calculate_max_initial_pressure(
                pipe_material,
                pipe_schedule,
                pipe_nps,
                welded,
                desired_fs,
                initial_temperature,
                species_dict,
                mechanism,
                max_pressure=max_pressure,
                error_tol=error_tol
            )

            states = tube.calculate_reflected_shock_state(
                test_result,
                initial_temperature,
                species_dict,
                mechanism
            )

            # get dynamic load factor
            dlf = tube.get_pipe_dlf(
                pipe_material,
                pipe_schedule,
                pipe_nps,
                states['cj']['speed']
            )

            calc_max = states['reflected']['state'].P
            max_solution = max_solution.to('Pa').magnitude / dlf

            error = abs(max_solution - calc_max) / max_solution

            assert error <= 0.0005
