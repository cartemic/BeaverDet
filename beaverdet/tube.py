# -*- coding: utf-8 -*-
"""
PURPOSE:
    A series of tools to aid in the design of a detonation tube.

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import warnings
import os
from math import sqrt
import pint
import pandas as pd
import numpy as np
import sympy as sp
import cantera as ct
from . import tools
from . import thermochem


class Bolt:
    @classmethod
    def calculate_stress_areas(
            cls,
            thread_size,
            thread_class,
            bolt_max_tensile,
            plate_max_tensile,
            engagement_length,
            unit_registry
    ):
        """
        Calculates internal and external thread stress areas using formulas in
        Machinery's Handbook, 26th edition.

        Parameters
        ----------
        thread_size : str
            Size of threads to be evaluated, e.g. '1/4-20' or '1 1/2-6'
        thread_class : str
            Class of threads to be evaluated, '2' or '3'. 'A' or 'B' are
            automatically appended for internal/external threads
        bolt_max_tensile : pint quantity
            Pint quantity of bolt (ext. thread) tensile failure stress
        plate_max_tensile : pint quantity
            Pint quantity of plate (int. thread) tensile failure stress
        engagement_length : pint quantity
            Pint quantity of total thread engagement length
        unit_registry : pint unit registry
            Unit registry for managing units to prevent conflicts with parent
            unit registry

        Returns
        -------
        thread : dict
            Dictionary with the following key/value pairs:
            'plate area': stress area of internal threads within the plate
            'screw area': stress area of external threads on the screw
            'minimum engagement': minimum engagement length causing screw to
                fail in tension rather than shear, thus preventing the plate
                from stripping.
        """
        quant = unit_registry.Quantity

        tools.check_pint_quantity(
            bolt_max_tensile,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            plate_max_tensile,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            engagement_length,
            'length',
            ensure_positive=True
        )

        # convert to local unit registry
        bolt_max_tensile = quant(
            bolt_max_tensile.magnitude,
            bolt_max_tensile.units.format_babel()
        )
        plate_max_tensile = quant(
            plate_max_tensile.magnitude,
            plate_max_tensile.units.format_babel()
        )
        engagement_length = quant(
            engagement_length.magnitude,
            engagement_length.units.format_babel()
        )

        thread = dict()

        # look up thread specs for stress area calculations
        thread_specs = cls._import_thread_specs()
        k_n_max = quant(
            thread_specs['internal']
            ['minor diameter max']
            [thread_size]
            [thread_class + 'B'],
            'in'
        )
        e_s_min = quant(
            thread_specs['external']
            ['pitch diameter min']
            [thread_size]
            [thread_class + 'A'],
            'in'
        )
        e_n_max = quant(
            thread_specs['internal']
            ['pitch diameter max']
            [thread_size]
            [thread_class + 'B'],
            'in'
        )
        d_s_min = quant(
            thread_specs['external']
            ['major diameter min']
            [thread_size]
            [thread_class + 'A'],
            'in'
        )
        tpi = quant(
            float(thread_size.split('-')[-1]),
            '1/in'
        )
        basic_diameter = quant(
            thread_specs['external']
            ['basic diameter']
            [thread_size]
            [thread_class + 'A'],
            'in'
        )

        if bolt_max_tensile < quant(100000, 'psi'):
            # calculate screw tensile area using eq. 9 (p. 1482) in Fasteners
            # section of Machinery's Handbook 26 (also eq. 2a on p. 1490)
            screw_area_tensile = np.pi / 4 * (
                    basic_diameter - 0.9742785 / tpi
            ) ** 2
        else:
            # calculate screw tensile area using eq. 2b (p. 1490) in Fasteners
            # section of Machinery's Handbook 26
            screw_area_tensile = np.pi * (
                    e_s_min / 2 -
                    0.16238 / tpi
            ) ** 2

        # calculate screw shear area using eq. 5 (p. 1491) in Fasteners section
        # of Machinery's Handbook 26
        screw_area_shear = (
                np.pi * tpi * engagement_length * k_n_max *
                (1. / (2 * tpi) + 0.57735 * (e_s_min - k_n_max))
        )

        # choose correct area
        if screw_area_shear < screw_area_tensile:
            warnings.warn(
                'Screws fail in shear, not tension.' +
                ' Plate may be damaged.' +
                ' Consider increasing bolt engagement length',
                Warning
            )
            thread['screw area'] = screw_area_shear
        else:
            thread['screw area'] = screw_area_tensile

        # calculate plate shear area using eq. 6 (p. 1491) in Fasteners section
        # of Machinery's Handbook 26
        thread['plate area'] = (
                np.pi * tpi * engagement_length * d_s_min *
                (1. / (2 * tpi) + 0.57735 * (d_s_min - e_n_max))
        )

        # calculate minimum engagement scale factor using eq. 3 (p. 1490) in
        # Fasteners section of Machinery's Handbook 26
        j_factor = (
                (screw_area_shear * bolt_max_tensile) /
                (thread['plate area'] * plate_max_tensile)
        )

        # calculate minimum thread engagement (corrected for material
        # differences) using eqs. 1 and 4 (pp. 1490-1491) in Fasteners section
        # of Machinery's Handbook 26
        thread['minimum engagement'] = (
            2 * screw_area_tensile / (
                k_n_max * np.pi * (
                    1. / 2 + 0.57735 * tpi * (e_s_min - k_n_max)
                    )
                )
            ) * j_factor

        return thread

    @staticmethod
    def _import_thread_specs():
        """
        Imports thread specifications from .csv files

        Returns
        -------
        thread_specs : dict
            [internal thread specs, external thread specs]. Both sets of thread
            specifications are multi-indexed with (thread size, thread class).
        """
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            'lookup_data'
        )
        file_names = [
            'ANSI_inch_internal_thread.csv',
            'ANSI_inch_external_thread.csv'
        ]
        file_locations = [
            os.path.relpath(
                os.path.join(
                    file_directory,
                    name
                )
            )
            for name in file_names
        ]

        thread_specs = {
            key: pd.read_csv(location, index_col=(0, 1)) for location, key in
            zip(file_locations, ['internal', 'external'])
        }

        return thread_specs

    @classmethod
    def get_thread_property(
            cls,
            thread_property,
            thread_size,
            thread_class,
            unit_registry
    ):
        """
        Finds a thread property, such as minor diameter, using a dataframe from
        import_thread_specs(). import_thread_specs is not directly called here
        to save time by not reading from disk every time a property is
        requested.

        Parameters
        ----------
        thread_property : str
            Property that is desired, such as 'minor diameter'
        thread_size : str
            Thread size for desired property, such as '1/4-20' or '1 1/2-6'
        thread_class : str
            Thread class: '2B' or '3B' for internal threads, '2A' or '3A' for
            external threads
        unit_registry : pint unit registry
            Unit registry for managing units to prevent conflicts with parent
            unit registry

        Returns
        -------
        pint.UnitRegistry().Quantity
            Property requested, as a pint quantity with units of inches
        """
        quant = unit_registry.Quantity
        thread_specs = cls._import_thread_specs()

        # determine if internal or external
        if 'A' in thread_class and ('2' in thread_class or '3' in thread_class):
            thread_specs = thread_specs['external']
        elif 'B' in thread_class and ('2' in thread_class
                                      or '3' in thread_class):
            thread_specs = thread_specs['internal']
        else:
            raise ValueError('\nbad thread class')

        # ensure property is in the specs dataframe
        if thread_property not in thread_specs.keys():
            raise KeyError('\nThread property \'' +
                           thread_property +
                           '\' not found. Available specs: ' +
                           "'" + "', '".join(thread_specs.keys()) + "'")

        # ensure thread size is in the specs dataframe
        if thread_size not in thread_specs.index:
            raise KeyError('\nThread size \'' +
                           thread_size +
                           '\' not found')

        # retrieve the property
        return quant(thread_specs[thread_property][thread_size][thread_class],
                     'in')


class DDT:
    @staticmethod
    def calculate_spiral_diameter(
            pipe_id,
            blockage_ratio
    ):
        """
        Calculates the diameter of a Shchelkin spiral corresponding to a given
        blockage ratio within a pipe of given inner diameter.

        Parameters
        ----------
        pipe_id : pint quantity
            Length scale representing the inner diameter of the pipe used for
            the detonation tube
        blockage_ratio : float
            percentage (float between 0 and 1)

        Returns
        -------
        spiral_diameter : pint quantity
            Shchelkin spiral diameter inside a tube of pipe_id inner diameter
            giving a blockage ratio of blockage_ratio %. Units are the same as
            pipe_id.
        """
        # ensure blockage ratio is a float
        try:
            blockage_ratio = float(blockage_ratio)
        except ValueError:
            raise ValueError('\nNon-numeric blockage ratio.')

        # ensure blockage ratio is on 0<BR<1
        if not 0 < blockage_ratio < 1:
            raise ValueError('\nBlockage ratio outside of 0<BR<1')

        tools.check_pint_quantity(
            pipe_id,
            'length',
            ensure_positive=True
        )

        # calculate Shchelkin spiral diameter
        spiral_diameter = pipe_id / 2 * (1 - sqrt(1 - blockage_ratio))
        return spiral_diameter

    @staticmethod
    def calculate_blockage_ratio(
            tube_inner_diameter,
            blockage_diameter
    ):
        """
        Calculates the blockage ratio of a Shchelkin spiral within a detonation
        tube.

        Parameters
        ----------
        tube_inner_diameter : pint quantity
            Length scale corresponding to the ID of the detonation tube
        blockage_diameter : pint quantity
            Length scale corresponding to the OD of a Shchelkin spiral

        Returns
        -------
        blockage_ratio : float
            Ratio of blocked to open area (between 0 and 1)
        """

        # check dimensionality and >=0
        tools.check_pint_quantity(
            tube_inner_diameter,
            'length',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            blockage_diameter,
            'length',
            ensure_positive=True
        )

        # make sure units cancel
        blockage_diameter = blockage_diameter.to_base_units()
        tube_inner_diameter = tube_inner_diameter.to_base_units()

        # ensure blockage diameter < tube diameter
        if tube_inner_diameter.magnitude == 0:
            raise ValueError('\ntube ID cannot be 0')
        elif blockage_diameter >= tube_inner_diameter:
            raise ValueError('\nblockage diameter >= tube diameter')

        # calculate blockage ratio
        blockage_ratio = (1 - (1 - 2 * blockage_diameter.magnitude /
                               tube_inner_diameter.magnitude) ** 2)

        return blockage_ratio

    @staticmethod
    def calculate_run_up(
            blockage_ratio,
            tube_diameter,
            initial_temperature,
            initial_pressure,
            species_dict,
            mechanism,
            unit_registry,
            phase_specification=''
    ):
        """
        Calculates the runup distance needed for a detonation to develop from a
        deflagration for a given blockage ratio, tube diameter, and mixture.
        This is accomplished using equations collected by Ciccarelli and
        Dorofeev [1] for blockage ratios <= 0.75. If the desired blockage ratio
        is less than 0.3, the mixture viscosity is needed, and the
        phase_specification option may be necessary depending on the mechanism.

        [1] G. Ciccarelli and S. Dorofeev, “Flame acceleration and transition to
        detonation in ducts,” Prog. Energy Combust. Sci., vol. 34, no. 4, pp.
        499–550, Aug. 2008.

        Parameters
        ----------
        blockage_ratio : float
            Ratio of the cross-sectional area of the detonation tube and a
            periodic blockage used to cause DDT
        tube_diameter : pint quantity
            Internal diameter of the detonation tube
        initial_temperature : pint quantity
            Mixture initial temperature
        initial_pressure : pint quantity
            Mixture initial pressure
        species_dict : dict
            Dictionary containing the species in the mixture as keys, with total
            moles or mole fractions as values
        mechanism : str
            Mechanism file name for Cantera
        unit_registry : pint unit registry
            Unit registry for managing units to prevent conflicts with parent
            unit registry
        phase_specification : str
            (Optional) Phase specification within the mechanism file used to
            evaluate thermophysical properties. If Gri30.cti is used with no
            phase specification, viscosity calculations will fail, resulting in
            an error for all blockage ratios less than 0.3.

        Returns
        -------
        runup_distance : pint quantity
            Predicted DDT distance, with the same units as the tube diameter
        """

        if blockage_ratio <= 0 or blockage_ratio > 0.75:
            raise ValueError('\nBlockage ratio outside of correlation range')

        tools.check_pint_quantity(
            tube_diameter,
            'length',
            ensure_positive=True
        )

        tools.check_pint_quantity(
            initial_temperature,
            'temperature',
            ensure_positive=True
        )

        tools.check_pint_quantity(
            initial_pressure,
            'pressure',
            ensure_positive=True
        )

        # handle units
        quant = unit_registry.Quantity
        tube_diameter = quant(
            tube_diameter.magnitude,
            tube_diameter.units.format_babel()
        )

        # calculate laminar flamespeed
        laminar_fs = thermochem.calculate_laminar_flamespeed(
            initial_temperature,
            initial_pressure,
            species_dict,
            mechanism
        )
        laminar_fs = quant(
            laminar_fs.magnitude, laminar_fs.units.format_babel()
        )

        # calculate density ratio across the deflagration assuming adiabatic
        # flame
        density = np.zeros(2)
        working_gas = ct.Solution(mechanism, phase_specification)
        working_gas.TPX = [
            initial_temperature.to('K').magnitude,
            initial_pressure.to('Pa').magnitude,
            species_dict
        ]
        density[0] = working_gas.density
        working_gas.equilibrate('HP')
        density[1] = 1 / working_gas.density
        density_ratio = np.prod(density)

        # find sound speed in products at adiabatic flame temperature
        sound_speed = thermochem.get_eq_sound_speed(
            quant(working_gas.T, 'K'),
            quant(working_gas.P, 'Pa'),
            species_dict,
            mechanism
        )
        sound_speed = quant(
            sound_speed.magnitude,
            sound_speed.units.format_babel()
        )

        def eq4_1():
            """
            Calculate runup distance for blockage ratios <= 0.1 using equation
            4.1 from G. Ciccarelli and S. Dorofeev, “Flame acceleration and
            transition to detonation in ducts,” Prog. Energy Combust. Sci.,
            vol. 34, no. 4, pp. 499–550, Aug. 2008.
            """
            # define constants
            kappa = 0.4
            kk = 5.5
            cc = 0.2
            mm = -0.18
            eta = 2.1

            # calculate laminar flame thickness, delta
            working_gas.TPX = [
                initial_temperature.to('K').magnitude,
                initial_pressure.to('Pa').magnitude,
                species_dict
            ]
            rho = quant(working_gas.density_mass, 'kg/m^3')
            mu = quant(working_gas.viscosity, 'Pa*s')
            nu = mu / rho
            delta = (nu / laminar_fs).to_base_units()

            # calculate gamma
            gamma = (
                            sound_speed /
                            (eta * (density_ratio - 1) ** 2 * laminar_fs) *
                            (delta / tube_diameter) ** (1. / 3)
                    ) ** (1 / (2 * mm + 7. / 3))

            # calculate runup distance
            d_over_h = (2. / (1 - np.sqrt(1 - blockage_ratio)))
            runup = (
                    gamma / cc *
                    (1 / kappa * np.log(gamma * d_over_h) + kk) *
                    tube_diameter
            )
            return runup.to(tube_diameter.units.format_babel())

        def eq4_4():
            """
            Calculate runup for blockage ratios between 0.3 and 0.75 using
            equation 4.4 in G. Ciccarelli and S. Dorofeev, “Flame acceleration
            and transition to detonation in ducts,” Prog. Energy Combust. Sci.,
            vol. 34, no. 4, pp. 499–550, Aug. 2008.
            """
            # define constants
            aa = 2.
            bb = 1.5

            # calculate left and right hand sides of eq 4.4
            lhs = (
                    2 * 10 * laminar_fs * (density_ratio - 1) /
                    (sound_speed * tube_diameter)
            )
            rhs = (
                    aa * (1 - blockage_ratio) /
                    (1 + bb * blockage_ratio)
            )

            runup = rhs / lhs

            return runup.to(tube_diameter.units.format_babel())

        # use appropriate equation to calculate runup distance
        if 0.3 <= blockage_ratio <= 0.75:
            runup_distance = eq4_4()
        elif 0.1 >= blockage_ratio:
            runup_distance = eq4_1()
        else:
            interp_distances = np.array([
                eq4_1().magnitude,
                eq4_4().magnitude
            ])
            runup_distance = np.interp(
                blockage_ratio,
                np.array([0.1, 0.3]),
                interp_distances
            )
            runup_distance = quant(
                runup_distance,
                tube_diameter.units.format_babel()
            )

        return runup_distance


class Window:
    @classmethod
    def safety_factor(
            cls,
            length,
            width,
            thickness,
            pressure,
            rupture_modulus
    ):
        """
        This function calculates the safety factor of a clamped rectangular
        window given window dimensions, design pressure, and material rupture
        modulus

        Parameters
        ----------
        length : pint quantity with length units
            Window unsupported (viewing) length
        width : pint quantity with length units
            Window unsupported (viewing) width
        thickness : pint quantity with length units
            Window thickness
        pressure : pint quantity with pressure units
            Design pressure differential across window at which factor of
            safety is to be calculated
        rupture_modulus : pint quantity with pressure units
            Rupture modulus of desired window material.

        Returns
        -------
        safety_factor : float
            Window factor of safety
        """

        tools.check_pint_quantity(
            length,
            'length',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            width,
            'length',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            thickness,
            'length',
            ensure_positive=True
        )

        tools.check_pint_quantity(
            pressure,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            rupture_modulus,
            'pressure',
            ensure_positive=True
        )

        safety_factor = cls.solver(
            length=length.to_base_units().magnitude,
            width=width.to_base_units().magnitude,
            thickness=thickness.to_base_units().magnitude,
            pressure=pressure.to_base_units().magnitude,
            rupture_modulus=rupture_modulus.to_base_units().magnitude
        )

        return safety_factor

    @classmethod
    def minimum_thickness(
            cls,
            length,
            width,
            safety_factor,
            pressure,
            rupture_modulus,
            unit_registry
    ):
        """
        This function calculates the thickness of a clamped rectangular window
        which gives the desired safety factor.

        Parameters
        ----------
        length : pint quantity with length units
            Window unsupported (viewing) length
        width : pint quantity with length units
            Window unsupported (viewing) width
        safety_factor : float
            Safety factor
        pressure : pint quantity with pressure units
            Design pressure differential across window at which factor of
            safety is to be calculated
        rupture_modulus : pint quantity with pressure units
            Rupture modulus of desired window material.
        unit_registry : pint unit registry
            Keeps output consistent with parent registry, avoiding conflicts

        Returns
        -------
        thickness : pint quantity
            Window thickness
        """
        quant = unit_registry.Quantity

        tools.check_pint_quantity(
            length,
            'length',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            width,
            'length',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            pressure,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            rupture_modulus,
            'pressure',
            ensure_positive=True
        )

        # Ensure safety factor is numeric and > 1
        try:
            if safety_factor < 1:
                raise ValueError('\nWindow safety factor < 1')
        except TypeError:
            raise TypeError('\nNon-numeric window safety factor')

        thickness = cls.solver(
            length=length.to_base_units().magnitude,
            width=width.to_base_units().magnitude,
            safety_factor=safety_factor,
            pressure=pressure.to_base_units().magnitude,
            rupture_modulus=rupture_modulus.to_base_units().magnitude
        )

        return quant(
            thickness,
            width.to_base_units().units).to(width.units.format_babel())

    @staticmethod
    def solver(
            **kwargs
    ):
        """
        This function uses sympy to solve for a missing window measurement.
        Inputs are five keyword arguments, with the following possible values:
            length
            width
            thickness
            pressure
            rupture_modulus
            safety_factor
        All of these arguments should be floats, and dimensions should be
        consistent (handling should be done in other functions, such as
        calculate_window_sf().

        Equation from:
        https://www.crystran.co.uk/userfiles/files/
        design-of-pressure-windows.pdf

        Parameters
        ----------
        kwargs

        Returns
        -------
        missing value as a float, or NaN if the result is imaginary
        """

        # Ensure that 5 keyword arguments were given
        if kwargs.__len__() != 5:
            raise ValueError('\nIncorrect number of arguments sent to solver')

        # Ensure all keyword arguments are correct
        good_arguments = [
            'length',
            'width',
            'thickness',
            'pressure',
            'rupture_modulus',
            'safety_factor'
        ]
        bad_args = []
        for arg in kwargs:
            if arg not in good_arguments:
                bad_args.append(arg)

        if len(bad_args) > 0:
            error_string = '\nBad keyword argument:'
            for arg in bad_args:
                error_string += '\n' + arg

            raise ValueError(error_string)

        # Define equation to be solved
        k_factor = 0.75  # clamped window factor
        argument_symbols = {
            'length': 'var_l',
            'width': 'var_w',
            'thickness': 'var_t',
            'pressure': 'var_p',
            'rupture_modulus': 'var_m',
            'safety_factor': 'var_sf'
        }
        var_l = sp.Symbol('var_l')
        var_w = sp.Symbol('var_w')
        var_t = sp.Symbol('var_t')
        var_p = sp.Symbol('var_p')
        var_m = sp.Symbol('var_m')
        var_sf = sp.Symbol('var_sf')
        expr = (
                var_l *
                var_w *
                sp.sqrt(
                    (
                            var_p *
                            k_factor *
                            var_sf /
                            (
                                    2 *
                                    var_m *
                                    (
                                            var_l ** 2 +
                                            var_w ** 2
                                    )
                            )
                    )
                ) - var_t
        )

        # Solve equation
        for arg in kwargs:
            expr = expr.subs(argument_symbols[arg], kwargs[arg])

        solution = sp.solve(expr)[0]

        if solution.is_real:
            return float(solution)
        else:
            warnings.warn('Window inputs resulted in imaginary solution.')
            return np.NaN

    @staticmethod
    def calculate_bolt_sfs(
            max_pressure,
            window_area,
            num_bolts,
            thread_size,
            thread_class,
            bolt_max_tensile,
            plate_max_tensile,
            engagement_length,
            unit_registry
    ):
        """
        Calculates bolt and plate safety factors for viewing window bolts

        Parameters
        ----------
        max_pressure : pint quantity
            Pint quantity of tube maximum pressure (absolute)
        window_area : pint quantity
            Pint quantity of window area exposed to high pressure environment
        num_bolts : int
            Number of bolts used to secure each viewing window
        thread_size : str
            Size of threads to be evaluated, e.g. '1/4-20' or '1 1/2-6'
        thread_class : str
            Class of threads to be evaluated, '2' or '3'. 'A' or 'B' are
            automatically appended for internal/external threads
        bolt_max_tensile : pint quantity
            Pint quantity of bolt (ext. thread) tensile failure stress
        plate_max_tensile : pint quantity
            Pint quantity of plate (int. thread) tensile failure stress
        engagement_length : pint quantity
            Pint quantity of total thread engagement length
        unit_registry : pint unit registry
            Keeps output consistent with parent registry, avoiding conflicts

        Returns
        -------
        safety_factors : dict
            Dictionary with keys of 'bolt' and 'plate', giving factors of safety
            for window bolts and the plate that they are screwed into.
        """
        quant = unit_registry.Quantity

        tools.check_pint_quantity(
            max_pressure,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            window_area,
            'area',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            bolt_max_tensile,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            plate_max_tensile,
            'pressure',
            ensure_positive=True
        )
        tools.check_pint_quantity(
            engagement_length,
            'length',
            ensure_positive=True
        )

        # convert all quantities to local unit registry
        max_pressure = quant(
            max_pressure.magnitude,
            max_pressure.units.format_babel()
        )
        window_area = quant(
            window_area.magnitude,
            window_area.units.format_babel()
        )
        bolt_max_tensile = quant(
            bolt_max_tensile.magnitude,
            bolt_max_tensile.units.format_babel()
        )
        plate_max_tensile = quant(
            plate_max_tensile.magnitude,
            plate_max_tensile.units.format_babel()
        )
        engagement_length = quant(
            engagement_length.magnitude,
            engagement_length.units.format_babel()
        )

        # get total force per bolt
        window_force = (
                (max_pressure - quant(1, 'atm')) * window_area / num_bolts
        )

        # get stress areas
        thread = Bolt.calculate_stress_areas(
            thread_size,
            thread_class,
            bolt_max_tensile,
            plate_max_tensile,
            engagement_length,
            unit_registry
        )
        screw_area = thread['screw area']
        screw_area = quant(
            screw_area.magnitude,
            screw_area.units.format_babel()
        )
        plate_area = thread['plate area']
        plate_area = quant(
            plate_area.magnitude,
            plate_area.units.format_babel()
        )

        # calculate safety factors
        safety_factors = dict()
        safety_factors['bolt'] = (
                bolt_max_tensile / (window_force / screw_area)
        ).to_base_units()
        safety_factors['plate'] = (
                plate_max_tensile / (window_force / plate_area)
        ).to_base_units()
        return safety_factors


class Tube:
    _all_quantities = {
        'material',
        'schedule',
        'nominal_size',
        'welded',
        'initial_pressure',
        'initial_temperature',
        'max_pressure',
        'max_stress',
        'dynamic_load_factor',
        'dimensions',
        'fuel',
        'oxidizer',
        'diluent',
        'equivalence_ratio',
        'dilution_fraction',
        'dilution_mode',
        'mechanism',
        'safety_factor',
        'flange_class',
        'cj_speed',
        'dimensions',
        'verbose',
        'show_warnings'
    }

    def __init__(
            self,
            *,
            material='316L',
            schedule='80',
            nominal_size='6',
            welded=False,
            max_stress=None,
            initial_temperature=(20, 'degC'),
            max_pressure=None,
            mechanism='gri30.cti',
            fuel='CH4',
            oxidizer='O2:1, N2:3.76',
            diluent='N2',
            equivalence_ratio=1,
            dilution_fraction=0,
            dilution_mode='mole',
            safety_factor=4,
            verbose=False,
            show_warnings=True,
            autocalc_initial=False,
            use_multiprocessing=False
    ):
        """
        Parameters
        ----------
        """
        self._initializing = True

        # decide whether to allow automatic calculations
        self._calculate_stress = max_stress is not None
        self._calculate_max_pressure = max_pressure is not None
        self._autocalc_initial = bool(autocalc_initial)

        # build local unit registry
        self._units = self._UnitSystem()

        # initiate hidden dict of properties
        self._properties = dict()

        # define all non-input quantities as None
        inputs = locals()
        for item in self._all_quantities:
            self._properties[item] = None

        # decide on use of multiprocessing (requires __main__)
        self._use_multiprocessing = bool(use_multiprocessing)

        # determine whether or not the tube is welded
        self._properties['welded'] = bool(welded)

        # check materials list to make sure it's good
        # define and collect tube materials and groups
        self._collect_tube_materials()
        self._get_material_groups()
        self._check_materials_list()
        self._collect_material_limits()
        self._get_flange_limits_from_csv()
        self.material = material
        self._pipe_schedules_import()
        self._mechanisms = tools.find_mechanisms()

        # determine whether or not to report progress or issues to the user
        self.verbose = bool(verbose)
        self._show_warnings = bool(show_warnings)

        # initialize dimensions object and set nominal size and schedule
        self._properties['dimensions'] = self._Dimensions()
        self.nominal_size = nominal_size
        self.schedule = schedule

        # set initial temperature to 20 C if not defined
        # if initial_temperature is None:
        #     self._properties[
        #         'initial_temperature'
        #     ] = self._units.quant(20, 'degC')
        # else:
        self.initial_temperature = initial_temperature

        # set max stress
        if max_stress is not None:
            self._properties['max_stress'] = max_stress

            # keep the user's input
            self._calculate_stress = False
        else:
            # allow max stress to be recalculated
            self._calculate_stress = True

        # set safety factor
        self.safety_factor = safety_factor

        # set max pressure
        if max_pressure is not None:
            self.max_pressure = max_pressure
            # allow max pressure to be recalculated
            self._calculate_max_pressure = False

        else:
            # keep the user's input
            self._calculate_max_pressure = True

        # set mechanism and reactant mixture
        if mechanism is not None:
            self.mechanism = mechanism
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.diluent = diluent
        self.equivalence_ratio = equivalence_ratio
        self.dilution_mode = dilution_mode
        self.dilution_fraction = dilution_fraction
        self._initializing = False

        # start auto-calculation chain
        if self._calculate_stress:
            self.calculate_max_stress()
        elif self._calculate_max_pressure:
            self.calculate_max_pressure()

    class _UnitSystem:
        def __init__(
                self
        ):
            self.ureg = pint.UnitRegistry()
            self.quant = self.ureg.Quantity

    class _Dimensions:
        def __init__(self):
            self.inner_diameter = None
            self.outer_diameter = None
            self.wall_thickness = None

    def _pipe_schedules_import(self):
        # collect pipe schedules
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            'lookup_data'
        )
        file_name = 'pipe_schedules.csv'
        file_location = os.path.relpath(
            os.path.join(
                file_directory,
                file_name
            )
        )
        self._schedules = pd.read_csv(file_location, index_col=0)

    def _dimensions_lookup(self):
        try:
            available_sizes = list(
                self._schedules[self.schedule].dropna().to_dict().keys()
            )
        except KeyError:
            raise ValueError('\nPipe schedule not found')

        # ensure size exists
        if self.nominal_size not in available_sizes:
            raise ValueError(
                '\nNominal size not found for given pipe schedule'
            )

        # look up/calculate dimensions
        outer_diameter = self._schedules['OD'][self.nominal_size]
        wall_thickness = self._schedules[self.schedule][self.nominal_size]
        inner_diameter = outer_diameter - 2 * wall_thickness

        # assign values to self with units
        self.dimensions.outer_diameter = self._units.quant(
            outer_diameter,
            'in'
        )
        self.dimensions.inner_diameter = self._units.quant(
            inner_diameter,
            'in'
        )
        self.dimensions.wall_thickness = self._units.quant(
            wall_thickness,
            'in'
        )

        # recalculate stress and pressure
        if self._calculate_stress:
            self.calculate_max_stress()
        elif self._calculate_max_pressure:
            # this is elif because the change in max stress will trigger
            # recalculation of max pressure if required
            self.calculate_max_pressure()
        else:
            pass

    @property
    def autocalc_initial(self):
        """
        Determines whether or not to auto-calculate max initial pressure.
        Defaults to False because this calculation takes a bit of time.
        """
        return self._autocalc_initial

    @autocalc_initial.setter
    def autocalc_initial(
            self,
            auto
    ):
        self._autocalc_initial = bool(auto)

    @property
    def show_warnings(self):
        """
        Determines whether or not to show warnings.
        """
        return self._show_warnings

    @show_warnings.setter
    def show_warnings(
            self,
            show
    ):
        self._show_warnings = bool(show)

    @property
    def available_pipe_sizes(self):
        """
        A list of available pipe sizes for detonation tube construction. Cannot
        be set manually.
        """
        return list(self._schedules.index)

    @available_pipe_sizes.setter
    def available_pipe_sizes(
            self,
            _
    ):
        # the user doesn't need to update this, ignore their input
        raise PermissionError(
            '\nPipe sizes can not be set manually.'
        )

    @property
    def available_pipe_schedules(self):
        """
        A list of available pipe schedules for detonation tube construction for
        the current pipe size. Cannot be set manually.
        """
        schedules = self._schedules.loc[self.nominal_size].dropna().index[1:]
        return list(schedules)

    @available_pipe_schedules.setter
    def available_pipe_schedules(
            self,
            _
    ):
        # the user doesn't need to update this, ignore their input
        raise PermissionError(
            '\nPipe schedules can not be set manually.'
        )

    def _check_materials_list(
            self
    ):
        """
        Makes sure that the materials in materials_list.csv have stress limits
        and flange ratings. This function relies on _get_material_groups(), and
        either raises an error or returns True.

        Returns
        -------
        True
        """
        # collect files
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            'lookup_data'
        )
        my_files = os.listdir(file_directory)
        flange_ratings = [file for file in my_files if "flange" in file.lower()]
        stress_limits = [file for file in my_files if "stress" in file.lower()]

        # make sure things were actually loaded
        if not bool(flange_ratings + stress_limits):
            raise FileNotFoundError(
                '\nno files containing "flange" or "stress" found'
            )

        # initialize an error string and error indicator. Error string will be
        # used to aggregate errors in the list of available materials so that
        # all issues may be rectified simultaneously.
        error_string = '\n'
        has_errors = False

        # make sure all pipe material limits are either welded or seamless
        # other types are permitted, but will raise a warning
        for file in stress_limits:
            if ('welded' not in file.lower()) and (
                    'seamless' not in file.lower()) and self.show_warnings:
                # warn that something is weird
                warnings.warn(
                    file + 'does not indicate whether it is welded or seamless'
                )

            # check the first row of the file in question to extract the names
            # of the materials that it contains stress limits for
            file_location = os.path.join(
                file_directory,
                file
            )
            with open(file_location, 'r') as current_file:
                # read the first line, strip off carriage return, and split by
                # comma separators. Ignore first value, as this is temperature.
                materials = current_file.readline().strip().split(',')[1:]

                # check to make sure that each material in the list of available
                # materials has a stress limit curve for the current limit type
                for item in self._materials['Grade'].values.astype(str):
                    if item not in materials:
                        # a material is missing from the limits spreadsheet.
                        # indicate that an error has occurred, and add it to the
                        # error string.
                        error_string += 'Material ' + item + ' not found in ' \
                                        + file_location + '\n'
                        has_errors = True

        # find out which material groups need to be inspected
        groups = set()
        for group in self._materials['Group'].values.astype(str):
            groups.add(group.replace('.', '_'))

        # check folder to make sure the correct files exist
        for group in groups:
            if not any(rating.find(group) > 0 for rating in flange_ratings):
                # current group was not found in any of the files
                error_string += 'material group ' + group + ' not found' + '\n'
                has_errors = True

        # report all errors
        if has_errors:
            raise ValueError(error_string)

        return True

    def _get_material_groups(
            self
    ):
        """
        Collects materials and their associated ASME B16.5 material groups
        from a dataframe of material properties

        Returns
        -------
        groups_dict
        """
        grades = self._materials.Grade.values.astype(str)
        groups = self._materials.Group.values.astype(str)
        groups_dict = {}
        for [grade, group] in zip(grades, groups):
            groups_dict[grade] = group

        self._material_groups = groups_dict

    def _collect_tube_materials(
            self,
    ):
        """
        Reads in a csv file containing tube materials, their corresponding
        ASME B16.5 material groups, and selected material properties.

        Returns
        -------
        materials_dataframe : pd.DataFrame
            Dataframe of materials and their corresponding material groups and
            properties
        """
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            'lookup_data'
        )
        file_name = 'materials_list.csv'
        file_location = os.path.relpath(
            os.path.join(
                file_directory,
                file_name
            )
        )

        # read in csv and extract information
        if os.path.exists(file_location):
            try:
                materials_dataframe = pd.read_csv(file_location)
            except pd.errors.EmptyDataError:
                raise ValueError('\n' + file_name + ' is empty')

        else:
            # raise an exception if the file doesn't exist
            raise ValueError('\n' + file_name + ' does not exist')

        # apply units
        materials_dataframe.ElasticModulus = [
            self._units.quant(item, 'GPa') for item in
            materials_dataframe['ElasticModulus'].values
        ]
        materials_dataframe.Density = [
            self._units.quant(item, 'g/cm^3') for item in
            materials_dataframe['Density'].values
        ]

        self._materials = materials_dataframe

    @property
    def available_tube_materials(self):
        """
        A list of available materials for detonation tube construction. Cannot
        be set manually.
        """
        return list(self._materials['Grade'])

    @available_tube_materials.setter
    def available_tube_materials(
            self,
            _
    ):
        # the user doesn't need to update this, ignore their input
        raise PermissionError(
            '\nAvailable tube materials can not be set manually.'
        )

    def _get_flange_limits_from_csv(self):
        """
        Reads in flange pressure limits as a function of temperature for
        different pressure classes per ASME B16.5. Temperature is in Centigrade
        and pressure is in bar.
        """
        groups = ['2.1', '2.2', '2.3']
        self._flange_limits = {group: None for group in groups}

        for group in groups:
            # ensure group is valid
            file_group = str(group).replace('.', '_')
            file_directory = os.path.join(
                os.path.dirname(os.path.relpath(__file__)),
                'lookup_data')
            file_name = 'ASME_B16_5_flange_ratings_group_' + file_group + \
                        '.csv'
            file_location = os.path.relpath(
                os.path.join(file_directory, file_name)
            )
            if not os.path.exists(file_location):
                raise FileNotFoundError(
                    '\n' + file_location + 'not found'
                )

            # import the correct .csv file as a pandas dataframe
            flange_limits = pd.read_csv(file_location)

            # ensure all temperatures and pressures are floats
            new_data = pd.np.array([
                pd.to_numeric(flange_limits[column].values, errors='coerce')
                for column in flange_limits.columns
            ]).transpose()
            flange_limits = pd.DataFrame(
                columns=flange_limits.columns,
                data=new_data
            ).fillna(0)

            # make sure pressures are positive
            if not all(
                flange_limits.loc[
                    :,
                    flange_limits.columns != 'Temperature'
                ].fillna(0).values.flatten() >= 0
            ):
                raise ValueError('\nPressure less than zero.')

            # add units to temperature column
            flange_limits['Temperature'] = [
                self._units.quant(temp, 'degC') for temp in
                flange_limits['Temperature']
            ]

            # add units to pressure columns
            for key in flange_limits.keys():
                if key != 'Temperature':
                    pressures = []
                    for pressure in flange_limits[key]:
                        if pressure < 0:
                            pressures.append(np.NaN)
                        else:
                            pressures.append(self._units.quant(
                                float(pressure), 'bar')
                            )
                    flange_limits[key] = pressures

            self._flange_limits[group] = flange_limits

    def _collect_material_limits(self):
        """
        Loads in material limits from cav.
        """
        self._check_materials_list()

        # collect files
        file_directory = os.path.join(
            os.path.dirname(
                os.path.relpath(__file__)
            ),
            'lookup_data'
        )
        file_name = 'ASME_B31_1_stress_limits_'
        if self.welded:
            file_name += 'welded.csv'
        else:
            file_name += 'seamless.csv'
        file_location = os.path.join(
            file_directory,
            file_name
        )
        self._material_limits = pd.read_csv(file_location, index_col=0)

    def _get_pipe_stress_limits(self):
        material_limits = self._material_limits[self.material]

        # apply units
        limits = {
            'temperature': ('degF', []),
            'stress': ('ksi', [])
        }
        for temp, stress in material_limits.items():
            limits['temperature'][1].append(temp)
            limits['stress'][1].append(stress)

        return limits

    def _get_pipe_dlf(
            self,
            cj_vel,
            plus_or_minus=0.1
    ):
        """
        This function calculates the dynamic load factor by which a detonation
        tube's static analysis should be scaled in order to account for the
        tube's response to pressure transients. DLF is based on the work of
        Shepherd [1]. Since the limits of "approximately equal to" are not
        define we assume a default value of plus or minus ten percent, thus
        plus_or_minus=0.1.

        [1] Shepherd, J. E. (2009). Structural Response of Piping to
        Internal Gas Detonation. Journal of Pressure Vessel Technology,
        131(3), 031204. https://doi.org/10.1115/1.3089497

        Parameters
        ----------
        plus_or_minus : float
            Defines the band about the critical velocity which is considered
            "approximately equal to" -- the default value of 0.1 means plus
            or minus ten percent.

        Returns
        -------
        dynamic_load_factor : float
            Factor by which the tube's static maximum pressure should be
            de-rated to account for transient response to detonation waves.
        """
        if not (0 < plus_or_minus < 1):
            raise ValueError(
                '\nplus_or_minus factor not between 0 and 1'
            )

        # get material properties
        properties_dataframe = self._materials.set_index('Grade')
        elastic_modulus = (
            properties_dataframe['ElasticModulus'][self.material].to
            ('Pa')
        )
        density = (
            properties_dataframe['Density'][self.material].to
            ('kg/m^3')
        )
        poisson = properties_dataframe['Poisson'][self.material]

        # set geometry
        # first /2 for averaging
        # second /2 to to convert diameter to radius
        radius = (self.dimensions.outer_diameter +
                  self.dimensions.inner_diameter) / 2. / 2.

        # calculate critical velocity
        crit_velocity = (
                ((elastic_modulus ** 2 * self.dimensions.wall_thickness
                  ** 2) /
                 (3. * density ** 2 * radius ** 2 * (1. - poisson ** 2))
                 ) ** (1. / 4)
        )

        # set limits for 'approximately Vcrit'
        bounds = [
            crit_velocity * (1. + plus_or_minus),
            crit_velocity * (1. - plus_or_minus)
        ]

        if cj_vel < bounds[1]:
            dynamic_load_factor = 1
        elif cj_vel > bounds[0]:
            dynamic_load_factor = 2
        else:
            dynamic_load_factor = 4

        return dynamic_load_factor

    def _get_property(
            self,
            current_property
    ):
        """
        Looks up a property from the hidden property dictionary _all_quantities.

        Parameters
        ----------
        current_property

        Returns
        -------
        whichever property you asked for
        """
        if self._properties[current_property] is not None:
            return self._properties[current_property]
        else:
            if self.show_warnings:
                warnings.warn('{0} has not been defined.'
                              .format(current_property))
            return None

    def _set_property(
            self,
            current_property,
            value
    ):
        if current_property in Tube._all_quantities:
            if (
                    (self._properties[current_property] is None)
                    or
                    (isinstance(
                        value,
                        type(self._properties[current_property])
                    ))
            ):
                # if the quantity is currently None or the type is the same as
                # the previously set quantity, go ahead and set the new one
                self._properties[current_property] = value

            elif current_property == 'mechanism' and (
                    isinstance(value, str) or isinstance(value, dict)
            ):
                # an exception to the type matching rule is mechanism, which
                # can be either a str or dict
                self._properties[current_property] = value

            else:
                # the quantity is the wrong type, don't set it
                raise TypeError('\nWrong quantity type')

        else:
            # the quantity being set is not on the approved list
            raise ValueError(
                '\nBad quantity designator. Approved quantities are:\n' +
                '\n'.join(self._all_quantities)
            )

    @property
    def nominal_size(self):
        """
        The nominal pipe size (NPS) of the pipe used to construct the detonation
        tube. NPS should be set using a string, e.g.

        `mytube.nominal_size = '1/4'`

        for NPS-1/4 or

        `mytube.nominal_size = '6'`

        for NPS-6.
        """
        return self._get_property('nominal_size')

    @nominal_size.setter
    def nominal_size(
            self,
            nominal_size
    ):
        nominal_size = str(nominal_size)
        if nominal_size not in self.available_pipe_sizes:
            raise ValueError(
                '\n{0} is not a valid pipe size. '.format(nominal_size) +
                'For a list of available sizes, try \n' +
                '`mytube.available_pipe_sizes`'
            )
        self._set_property('nominal_size', nominal_size)
        if not self._initializing:
            self._dimensions_lookup()

    @property
    def schedule(self):
        """
        Pipe schedule as a string, e.g. '80', 'XXS'
        """
        return self._get_property('schedule')

    @schedule.setter
    def schedule(
            self,
            schedule
    ):
        schedule = str(schedule)
        if schedule not in self.available_pipe_schedules:
            raise ValueError(
                '\n{0} is not a valid pipe schedule for this nominal size. '
                .format(schedule) +
                'For a list of available schedules, try \n' +
                '`mytube.available_pipe_schedules`'
            )
        self._set_property('schedule', schedule)
        self._dimensions_lookup()

    @property
    def dimensions(self):
        """
        Cross-sectional dimensions of the pipe used to build the detonation
        tube.
        """
        return self._get_property('dimensions')

    @dimensions.setter
    def dimensions(
            self,
            _
    ):
        raise PermissionError(
            '\nTube dimensions are looked up based on nominal pipe size and ' +
            'schedule, not set. Try `mytube.schedule()` or ' +
            '`mytube.nominal_size()` instead.'
        )

    @property
    def material(self):
        """
        Material that pipe is made of as a string, e.g. '316L'
        """
        return self._get_property('material')

    @material.setter
    def material(
            self,
            material
    ):
        # make sure material is okay and store value
        material = str(material)
        if material not in self._materials['Grade'].values:
            raise ValueError('\nPipe material not found. For a list of '
                             'available materials try:\n'
                             '`mytube.available_tube_materials`')
        else:
            self._properties['material'] = material
            if self._calculate_stress and not self._initializing:
                self.calculate_max_stress()

    @property
    def welded(self):
        """
        True for welded pipe, False for seamless.
        """
        return self._get_property('welded')

    @welded.setter
    def welded(
            self,
            welded
    ):
        self._set_property('welded', bool(welded))

        # recalculate max stress
        if self._calculate_stress and not self._initializing:
            self.calculate_max_stress()

    @property
    def initial_temperature(self):
        """
        Initial temperature for reactant mixture. Can be set with either a pint
        quantity or an iterable of (temperature, 'units').
        """
        return self._get_property('initial_temperature')

    @initial_temperature.setter
    def initial_temperature(
            self,
            initial_temperature
    ):
        # check to see if input was an iterable
        initial_temperature = self._parse_quant_input(
            initial_temperature
        )

        # ensure temperature is a pint quantity, and convert it to the local
        # unit registry to avoid problems
        tools.check_pint_quantity(
            initial_temperature,
            'temperature',
            ensure_positive=True
        )

        self._set_property('initial_temperature', initial_temperature)
        if not self._initializing:
            if not self._calculate_stress:
                self.lookup_flange_class()
                if self.autocalc_initial:
                    self.calculate_initial_pressure()
            else:
                self.calculate_max_stress()
                if not self._calculate_max_pressure:
                    self.lookup_flange_class()
                else:
                    # flange lookup and initial pressure calc will spring from
                    # max pressure calculation
                    pass

    def _parse_quant_input(
            self,
            quant_input
    ):
        """
        Converts an iterable of (magnitude, 'units') to a pint quantity or
        converts a pint quantity to the local registry.

        Parameters
        ----------
        quant_input

        Returns
        -------
        input as a pint quantity
        """
        if hasattr(quant_input, 'magnitude'):
            return self._units.quant(
                quant_input.magnitude,
                quant_input.units.format_babel()
            )
        elif hasattr(quant_input, '__iter__'):
            if len(quant_input) != 2 and self.show_warnings:
                warnings.warn('too many arguments given, ignoring extras')

            return self._units.quant(float(quant_input[0]), quant_input[1])
        else:
            raise ValueError(
                'bad quantity input: {0}'.format(quant_input)
            )

    @property
    def safety_factor(self):
        """
        Desired safety factor for the detonation tube
        """
        return self._get_property('safety_factor')

    @safety_factor.setter
    def safety_factor(
            self,
            safety_factor
    ):
        self._set_property('safety_factor', float(safety_factor))

        if self._calculate_max_pressure:
            self.calculate_max_pressure()

    @property
    def max_stress(self):
        """
        Maximum allowable pipe stress. Can be set with either a pint quantity
        or an iterable of (max stress, 'units').
        """
        return self._get_property('max_stress')

    @max_stress.setter
    def max_stress(
            self,
            max_stress
    ):
        # check to see if input was an iterable
        max_stress = self._parse_quant_input(
            max_stress
        )

        # make sure input stress is a pint quantity with pressure units
        # and use it
        tools.check_pint_quantity(
            max_stress,
            'pressure',
            ensure_positive=True
        )
        self._set_property('max_stress', max_stress)

        # keep the user-input max stress
        self._calculate_stress = False

        if self._calculate_max_pressure:
            self.calculate_max_pressure()

    @property
    def max_pressure(self):
        """
        Maximum allowable pressure within the detonation tube, which should
        correspond to the reflection pressure. Can be set with either a pint
        quantity or an iterable of (max pressure, 'units').
        """
        return self._get_property('max_pressure')

    @max_pressure.setter
    def max_pressure(
            self,
            max_pressure
    ):
        # check to see if input was an iterable
        max_pressure = self._parse_quant_input(
            max_pressure
        )

        # make sure input pressure is a pint quantity with pressure units
        # and use it
        tools.check_pint_quantity(
            max_pressure,
            'pressure',
            ensure_positive=True
        )
        self._set_property('max_pressure', max_pressure)

        # keep the user-input max pressure
        self._calculate_max_pressure = False

        self.lookup_flange_class()
        if self.autocalc_initial:
            self.calculate_initial_pressure()

    @property
    def mechanism(self):
        """
        Mechanism file for Cantera solution objects.
        """
        return self._get_property('mechanism')

    @mechanism.setter
    def mechanism(
            self,
            mechanism
    ):
        if mechanism not in self._mechanisms:
            raise ValueError('\nMechanism not found. Available mechanisms:\n' +
                             '\n'.join(self._mechanisms))
        else:
            self._set_property('mechanism', mechanism)

            if not self._initializing:
                species = {'fuel': self.fuel,
                           'oxidizer': self.oxidizer,
                           'diluent': self.diluent}
                for component, item in species.items():
                    try:
                        self._check_species(item)
                    except ValueError:
                        if self.show_warnings:
                            warnings.warn(str(item) +
                                          ' not found in mechanism ' +
                                          self.mechanism +
                                          ', please define a new ' +
                                          component)
                        self._properties[component] = None
                        self._reactant_mixture = None

    def _check_species(
            self,
            species
    ):
        """
        Checks to make sure a species (fuel, oxidizer, diluent) is in the
        current mechanism.
        """
        gas = ct.Solution(self.mechanism)

        try:
            components = species.replace(' ', '').split(',')
            if len(components) > 1:
                # multiple components given, convert to a dict
                mixture = dict()
                for component in components:
                    current_species = component.split(':')
                    mixture[current_species[0]] = float(current_species[1])
            else:
                # single component as string
                mixture = {species: 1}

            gas.X = mixture

        except ct.CanteraError as err:
            err = str(err)
            start_loc = err.find('Unknown')
            end_loc = err.rfind('\n*******************************************'
                                '****************************\n')
            raise ValueError('\n' + err[start_loc:end_loc])

        except AttributeError:
            # this happens when a component is None, which occurs when the
            # mechanism is changed and the component is not in that mechanism.
            # Passing because the _get_property method already generates a
            # warning for this.
            pass

    @property
    def fuel(self):
        """
        Fuel to use in the reactant mixture, as a string. Must be included in
        the mechanism.
        """
        return self._get_property('fuel')

    @fuel.setter
    def fuel(
            self,
            fuel
    ):
        self._check_species(fuel)
        self._set_property('fuel', fuel)

        if not self._initializing:
            self._build_gas_mixture()

    @property
    def oxidizer(self):
        """
        Oxidizer to use in the reactant mixture, as a string. Must be included
        in the mechanism.
        """
        return self._get_property('oxidizer')

    @oxidizer.setter
    def oxidizer(
            self,
            oxidizer
    ):
        self._check_species(oxidizer)
        self._set_property('oxidizer', oxidizer)

        if not self._initializing:
            self._build_gas_mixture()

    @property
    def diluent(self):
        """
        Diluent to use in the reactant mixture, as a string. Must be included in
        the mechanism.
        """
        return self._get_property('diluent')

    @diluent.setter
    def diluent(
            self,
            diluent
    ):
        self._check_species(diluent)
        self._set_property('diluent', diluent)

        if not self._initializing:
            self._build_gas_mixture()

    @property
    def equivalence_ratio(self):
        """
        Desired reactant equivalence ratio. Cantera treats equivalence ratios
        less than or equal to zero as 100% oxidizer.
        """
        return self._get_property('equivalence_ratio')

    @equivalence_ratio.setter
    def equivalence_ratio(
            self,
            equivalence_ratio
    ):
        # note: if equivalence is <=0, cantera makes the mixture 100% oxidizer
        equivalence_ratio = float(equivalence_ratio)
        self._set_property('equivalence_ratio', equivalence_ratio)

        if not self._initializing:
            self._build_gas_mixture()

    @property
    def dilution_mode(self):
        """
        method of dilution (mass or mole)
        """
        return self._get_property('dilution_mode')

    @dilution_mode.setter
    def dilution_mode(
            self,
            dilution_mode
    ):
        dilution_mode = str(dilution_mode)
        if any([dilution_mode == 'mole',
                dilution_mode == 'mol',
                dilution_mode == 'molar']):
            # kind of easy to screw this one up
            dilution_mode = 'mole'
        elif dilution_mode == 'mass':
            # explicit is better, leaving this in.
            pass
        else:
            raise ValueError('\nBad dilution mode. Please use \'mole\' or'
                             ' \'mass\'')

        self._set_property('dilution_mode', dilution_mode)

        if not self._initializing:
            self._build_gas_mixture()

    @property
    def dilution_fraction(self):
        """
        Fraction of gas to be made up of diluent, by either mass or mole
        depending on dilution_mode.
        """
        return self._get_property('dilution_fraction')

    @dilution_fraction.setter
    def dilution_fraction(
            self,
            dilution_fraction
    ):
        dilution_fraction = float(dilution_fraction)
        if dilution_fraction < 0 or dilution_fraction > 1:
            raise ValueError(
                '\ndilution fraction must be between 0 and 1'
            )
        self._set_property('dilution_fraction', dilution_fraction)
        self._build_gas_mixture()

    def _build_gas_mixture(self):
        """
        Builds a reactant gas mixture from fuel, oxidizer, diluent, equivalence
        ratio, dilution fraction, and dilution mode
        """
        # if any of the components are None, the mixture can't be built. The
        # case of diluent=None is handled during the dilution step, meaning
        # that a mixture with no diluent won't crash the whole system.
        if not any([
            self.fuel is None,
            self.oxidizer is None
        ]):
            # initialize gas object
            gas = ct.Solution(self.mechanism)

            # set equivalence ratio
            gas.set_equivalence_ratio(
                self.equivalence_ratio,
                self.fuel,
                self.oxidizer
            )

            def dilute():
                return '{0}: {1} {2}: {3} {4}: {5}'.format(
                    self.diluent,
                    self.dilution_fraction,
                    self.fuel,
                    new_fuel_fraction,
                    self.oxidizer,
                    new_oxidizer_fraction)

            # apply dilution
            if self.dilution_fraction > 0:
                if self.diluent is None:
                    raise ValueError(
                        '\nCannot dilute mixture, please define a diluent'
                    )
                else:
                    if self.dilution_mode == 'mole':
                        mole_fractions = gas.mole_fraction_dict()
                        new_fuel_fraction = (1 - self.dilution_fraction) * \
                            mole_fractions[self.fuel]
                        new_oxidizer_fraction = (1 - self.dilution_fraction) * \
                            mole_fractions[self.oxidizer]
                        gas.X = dilute()
                        self._reactant_mixture = gas.mole_fraction_dict()
                    else:
                        mass_fractions = gas.mass_fraction_dict()
                        new_fuel_fraction = (1 - self.dilution_fraction) * \
                            mass_fractions[self.fuel]
                        new_oxidizer_fraction = (1 - self.dilution_fraction) * \
                            mass_fractions[self.oxidizer]
                        gas.Y = dilute()
                        self._reactant_mixture = gas.mass_fraction_dict()
            else:
                # undiluted
                self._reactant_mixture = gas.mole_fraction_dict()

    @property
    def reactant_mixture(self):
        """
        Reactant mixture for detonations.
        """
        return self._reactant_mixture

    @reactant_mixture.setter
    def reactant_mixture(
            self,
            _
    ):
        raise PermissionError(
            '\nReactant mixture is automatically adjusted based on:\n'
            '    - fuel\n'
            '    - oxidizer\n'
            '    - equivalence ratio\n'
            '    - diluent\n'
            '    - dilution fraction\n'
            '    - dilution mode'
        )

    @property
    def initial_pressure(self):
        """
        Maximum initial pressure resulting in the desired safety factor.
        """
        return self._get_property('initial_pressure')

    @initial_pressure.setter
    def initial_pressure(
            self,
            _
    ):
        raise PermissionError(
            '\nInitial pressure must be calculated, not set. Try'
            ' `mytube.calculate_initial_pressure()` instead.'
        )

    @property
    def dynamic_load_factor(self):
        """
        Factor accounting for the transient tube response to the detonation, per
        J. Shepherd, "Structural Response of Piping to Internal Gas Detonation",
        Journal of Pressure Vessel Technology, vol. 131, issue 3, pp. 031204,
        2009
        """
        return self._get_property('dynamic_load_factor')

    @dynamic_load_factor.setter
    def dynamic_load_factor(
            self,
            _
    ):
        raise PermissionError(
            '\nDynamic load factor must be calculated, not set. DLF calculation'
            ' occurs during initial pressure calculations. Try'
            ' `mytube.calculate_initial_pressure()` instead.'
        )

    @property
    def cj_speed(self):
        """
        Chapman-Jouguet speed resulting from a detonation in the reactant
        mixture.
        """
        return self._get_property('cj_speed')

    @cj_speed.setter
    def cj_speed(
            self,
            _
    ):
        raise PermissionError(
            '\nCJ speed must be calculated, not set. CJ speed calculation'
            ' occurs during initial pressure calculations. Try'
            ' `mytube.calculate_initial_pressure()` instead.'
        )

    @property
    def flange_class(self):
        """
        Minimum safe flange class to be used in tube construction.
        """
        return self._get_property('flange_class')

    @flange_class.setter
    def flange_class(
            self,
            _
    ):
        raise PermissionError(
            '\nFlange class must be calculated looked up based on max pressure,'
            ' not set. Try `mytube.lookup_flange_class()` instead.'
        )

    @property
    def verbose(self):
        """
        If True, calculation status will be printed to the console.
        """
        return self._get_property('verbose')

    @verbose.setter
    def verbose(
            self,
            verbose
    ):
        if verbose:
            verbose = True
        else:
            verbose = False

        self._set_property('verbose', verbose)

    def calculate_max_stress(self):
        """
        Finds the maximum allowable stress of a tube material at the tube's
        initial temperature

        Returns
        -------
        max_stress : pint quantity
            Pint quantity of maximum allowable tube stress
        """
        # Requires: initial_temperature, welded, material
        # ---------------------------------------------------------------------
        # initial temperature has default on __init__
        # welded has default on __init__
        # material has default on __init__

        if self.verbose:
            print('calculating max stress... ', end='')

        # look up stress-temperature limits and units
        stress_limits = self._get_pipe_stress_limits()
        stress_units = stress_limits['stress'][0]
        stresses = stress_limits['stress'][1]
        temp_units = stress_limits['temperature'][0]
        temperatures = stress_limits['temperature'][1]

        # ensure material stress limits have monotonically increasing
        # temperatures, otherwise the np.interp "results are nonsense" per
        # scipy docs
        if not np.all(np.diff(temperatures) > 0):
            raise ValueError('\nStress limits require temperatures to be ' +
                             'monotonically increasing')

        # interpolate max stress
        # noinspection PyAttributeOutsideInit
        max_stress = self._units.quant(
            np.interp(
                self.initial_temperature.to(temp_units).magnitude,
                temperatures,
                stresses
            ),
            stress_units
        )

        if self.verbose:
            print('Done')

        # noinspection PyAttributeOutsideInit
        self.max_stress = max_stress

        # allow max stress to be recalculated automatically
        self._calculate_stress = True
        return max_stress

    def calculate_max_pressure(
            self
    ):
        """
        Calculates the maximum allowable pressure from the limits found in ASME
        B31.1, with the option to modify the safety factor via

        `mytube.safety_factor()`

        and returns it as a pint quantity.
        """

        # Requires: max stress, dimensions, safety factor
        # ---------------------------------------------------------------------
        # safety factor has default on __init__
        # dimensions are found from schedule and nominal, which have defaults
        # max stress is either set by the user or calc'd on __init__

        if self.verbose:
            print('calculating max pressure... ', end='')

        # get required quantities
        dimensions = self._properties['dimensions']
        max_stress = self._properties['max_stress']
        safety_factor = self._properties['safety_factor']

        # calculate max pressure using basic longitudinal joint formula
        # on page 14 of Megyesy's Pressure Vessel Handbook, 8th ed.
        mean_diameter = (dimensions.outer_diameter +
                         dimensions.inner_diameter) / 2.
        asme_fs = 4
        max_pressure = (
                max_stress * (2 * dimensions.wall_thickness) *
                asme_fs / (mean_diameter * safety_factor)
        )

        if self.verbose:
            print('Done')

        self.max_pressure = max_pressure

        # allow max pressure to be recalculated automatically
        self._calculate_max_pressure = True
        return max_pressure

    def calculate_initial_pressure(
            self,
            error_tol=1e-4,
            max_iterations=500
    ):
        """
        Parameters
        ----------
        error_tol : float
            Relative error tolerance below which initial pressure calculations
            are considered 'good enough'
        max_iterations : int
            Maximum number of loop iterations before exit, defaults to 500

        Returns
        -------
        initial_pressure : pint quantity
            Initial mixture pressure corresponding to the tube's maximum
            allowable pressure.
        """
        # Requires: reactant_mixture, mechanism, max_pressure, and
        # initial_temperature
        # ---------------------------------------------------------------------
        # reactant mixture is set from args with defaults on __init__
        # mechanism has default on __init__
        # max pressure is either set by user or calc'd on __init__
        # initial temperature is either set by user or calc'd on __init__

        if self.verbose:
            print('calculating initial pressure...')

        # get a rough estimate of the initial pressure
        # CJ pressure is approximately 17.5 times initial
        # reflected pressure is approximately 2.5 times CJ
        # worst case dynamic load factor is 4
        p_max = self.max_pressure.to('Pa')
        # initial_pressure = p_max / (17.5 * 2.5 * 4)
        initial_pressure = self._units.quant(1, 'atm')
        counter = 0
        error_tol = abs(error_tol)

        if self.verbose:
            print('    calculating reflected shock state... ', end='')
        state = thermochem.calculate_reflected_shock_state(
            self.initial_temperature,
            initial_pressure,
            self.reactant_mixture,
            self.mechanism,
            self._units.ureg,
            use_multiprocessing=self._use_multiprocessing
        )
        dlf = self._get_pipe_dlf(state['cj']['speed'])

        error = (initial_pressure.magnitude * dlf / p_max.magnitude) - 1.
        if self.verbose:
            print('  error: {0:1.3} %'.format(error * 100.))
        while abs(error) > error_tol and counter < max_iterations:
            counter += 1

            # update initial pressure guess
            initial_pressure = initial_pressure * p_max.magnitude / \
                (dlf * state['reflected']['state'].P)

            if self.verbose:
                print('    recalculating reflected shock state... ', end='')

            # get reflected shock pressure
            state = thermochem.calculate_reflected_shock_state(
                self.initial_temperature,
                initial_pressure,
                self.reactant_mixture,
                self.mechanism,
                self._units.ureg,
                use_multiprocessing=self._use_multiprocessing
            )

            # calculate new error, accounting for dynamic load factor
            dlf = self._get_pipe_dlf(state['cj']['speed'])
            error = (state['reflected']['state'].P * dlf - p_max.magnitude) \
                / p_max.magnitude
            if self.verbose:
                print('error: {0:1.3} %'.format(error * 100.))

        self._set_property('initial_pressure', initial_pressure)
        self._set_property('cj_speed',
                           self._units.quant(
                               state['cj']['speed'].to('m/s').magnitude,
                               'm/s'
                           ))
        self._set_property('dynamic_load_factor', dlf)

        if self.verbose:
            print('Done.')
        return initial_pressure

    def lookup_flange_class(
            self
    ):
        """
        Finds the minimum allowable flange class per ASME B16.5 for a give
        flange temperature and tube pressure.

        Returns
        -------
        flange_class: str
            String representing the minimum allowable flange class
        """
        # Requires: max_pressure, initial_temperature, material
        # ---------------------------------------------------------------------
        # max pressure is either set by user or calc'd on __init__
        # initial temperature is either set by user or calc'd on __init__
        # material has default on __init__

        if self.verbose:
            print('looking up flange class... ', end='')

        max_pressure = self.max_pressure
        initial_temperature = self.initial_temperature
        material = self.material

        # get ASME B16.5 material group
        group = self._material_groups[material]

        # import flange limits from csv
        flange_limits = self._flange_limits[group]

        # locate max pressure and convert to bar just in case
        class_keys = flange_limits.keys()[1:]
        max_key = '0'
        for key in class_keys:
            if int(key) > int(max_key):
                max_key = key
        max_ok_pressure = flange_limits[max_key].dropna().max()

        # ensure pressure is within bounds
        if max_pressure > max_ok_pressure:
            # pressure is outside of range, return an error
            raise ValueError('\nPressure out of range.')

        # locate max and min temperature and convert to degC just in case
        max_temp = flange_limits['Temperature'].max()
        min_temp = flange_limits['Temperature'].min()

        # ensure temperature is within bounds
        if ((initial_temperature < min_temp) or (
                initial_temperature > max_temp)):
            # temperature is outside of range, return an error
            raise ValueError('\nTemperature out of range.')

        # ensure class keys are sorted in rising order
        class_keys = sorted([(int(key), key) for key in class_keys])
        class_keys = [pair[1] for pair in class_keys]

        # find proper flange class
        correct_class = None
        for key in class_keys:
            max_class_pressure = flange_limits[key].dropna().max()
            if max_pressure < max_class_pressure:
                correct_class = int(key)
                break

        self._set_property('flange_class', correct_class)
        if self.verbose:
            print('Done')
        return correct_class
