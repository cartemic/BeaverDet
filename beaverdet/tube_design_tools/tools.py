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


import os
from math import sqrt
import pint
import pandas as pd
import numpy as np
import cantera as ct
from . import accessories as acc


def get_flange_limits_from_csv(
        group=2.3
):
    """
    Reads in flange pressure limits as a function of temperature for different
    pressure classes per ASME B16.5. Temperature is in Centigrade and pressure
    is in bar.

    Inputs:
        group: float or string of ASME B16.5 material group (defaults to 2.3).
             Only groups 2.1, 2.2, and 2.3 are included in the current release.

    Outputs:
        flange_limits: pandas dataframe, the first column of which is
             temperature. All other columns' keys are flange classes, and the
             values are the appropriate pressure limits in bar.
    """

    # ensure group is valid
    group = str(group).replace('.', '_')
    file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'lookup_data')
    file_name = 'ASME_B16_5_flange_ratings_group_' + group + '.csv'
    file_location = os.path.relpath(os.path.join(file_directory, file_name))

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity
    if os.path.exists(file_location):
        # import the correct .csv file as a pandas dataframe
        flange_limits = pd.read_csv(file_location)

        # ensure all temperatures and pressures are floats, and check to make
        # sure pressures are greater than zero
        values = flange_limits.values
        for row_number, row in enumerate(values):
            for column_number, item in enumerate(row):
                # ensure each item is a float and assign non-numeric values
                # a value of zero
                try:
                    values[row_number][column_number] = float(item)
                except ValueError:
                    values[row_number][column_number] = 0.

                if column_number > 0:
                    # these are pressures, which must be positive
                    if values[row_number][column_number] < 0:
                        raise ValueError('Pressure less than zero.')

        # add units to temperature column
        flange_limits['Temperature'] = [quant(temp, ureg.degC) for temp in
                                        flange_limits['Temperature']]

        # add units to pressure columns
        for key in flange_limits.keys():
            if key != 'Temperature':
                flange_limits[key] = [quant(pressure, ureg.bar) for pressure in
                                      flange_limits[key]]

        return flange_limits

    else:
        # the user gave a bad group label
        raise ValueError('{0} is not a valid group'.format(group))


def lookup_flange_class(
        temperature,
        pressure,
        desired_material
):
    """
    Finds the minimum allowable flange class per ASME B16.5 for a give flange
    temperature and tube pressure.

    Inputs:
        temperature: pint quantity with temperature units
        pressure: pint quantity with pressure units
        desired_material: string of desired flange material

    Outputs:
        flange_class: string representing the minimum allowable flange class
    """
    # ensure desired_material is a string
    if not isinstance(desired_material, str):
        raise ValueError('Desired material non-string input.')

    # read in available materials and their associated groups
    materials_dict = acc.get_material_groups()

    # ensure desired_material is in materials_dict
    if desired_material not in materials_dict.keys():
        raise ValueError('Desired material not in database.')
    else:
        # material is good, get ASME B16.5 material group
        group = materials_dict[desired_material]

    # initialize unit registry for unit handling
    ureg = pint.UnitRegistry()

    # type check to make sure temperature is a pint quantity
    acc.check_pint_quantity(
        temperature,
        'temperature'
    )

    # type check to make sure pressure is a pint quantity
    acc.check_pint_quantity(
        pressure,
        'pressure'
    )

    # import flange limits from csv
    flange_limits = get_flange_limits_from_csv(group)

    # locate max pressure and convert to bar just in case
    class_keys = flange_limits.keys()[1:]
    max_key = '0'
    for key in class_keys:
        if int(key) > int(max_key):
            max_key = key
    max_pressure = flange_limits[max_key].max().to(ureg.bar)

    # ensure pressure is within bounds
    if (pressure.magnitude < 0) or (pressure.magnitude >
                                    max_pressure.magnitude):
        # pressure is outside of range, return an error
        raise ValueError('Pressure out of range.')

    # locate max and min temperature and convert to degC just in case
    max_temp = flange_limits['Temperature'].max().to(ureg.degC)
    min_temp = flange_limits['Temperature'].min().to(ureg.degC)

    # ensure temperature is within bounds
    if (temperature.magnitude < min_temp.magnitude) or (temperature.magnitude >
                                                        max_temp.magnitude):
        # temperature is outside of range, return an error
        raise ValueError('Temperature out of range.')

    # ensure class keys are sorted in rising order
    class_keys = sorted([(int(key), key) for key in class_keys])
    class_keys = [pair[1] for pair in class_keys]

    # find proper flange class
    correct_class = None
    for key in class_keys:
        max_class_pressure = flange_limits[key].max().to(ureg.bar).magnitude
        if pressure.magnitude < max_class_pressure:
            correct_class = key
            break
    return correct_class


def calculate_spiral_diameter(
        pipe_id,
        blockage_ratio
):
    """
    Calculates the diameter of a Shchelkin spiral corresponding to a given
    blockage ratio within a pipe of given inner diameter.

    Inputs:
        pipe_id: pint quantity with a length scale representing the inner
            diameter of the pipe used for the detonation tube
        blockage_ratio: percentage (float between 0 and 100)

    Outputs:
        spiral_diameter: pint quantity representing the Shchelkin spiral
            diameter inside a tube of pipe_id inner diameter giving a blockage
            ratio of blockage_ratio %. Units are the same as pipe_id.
    """
    # ensure blockage ratio is a float
    try:
        blockage_ratio = float(blockage_ratio)
    except ValueError:
        raise ValueError('Non-numeric blockage ratio.')

    # ensure blockage ratio is on 0<BR<100
    if not 0 < blockage_ratio < 100:
        raise ValueError('Blockage ratio outside of 0<BR<100')

    acc.check_pint_quantity(
        pipe_id,
        'length',
        ensure_positive=True
    )

    # calculate Shchelkin spiral diameter
    spiral_diameter = pipe_id / 2 * (1 - sqrt(1 - blockage_ratio / 100))
    return spiral_diameter


def calculate_blockage_ratio(
        tube_inner_diameter,
        blockage_diameter
):
    """
    Calculates the blockage ratio of a Shchelkin spiral within a detonation
    tube.

    Inputs:
        tube_inner_diameter: pint quantity with a length scale corresponding
            to the ID of the detonation tube
        blockage_diameter: pint quantity with a length scale corresponding to
            the OD of a Shchelkin spiral

    Outputs:
        blockage_ratio: float between 0 and 100, representing the resulting
            blockage ratio in percent
    """

    # check dimensionality and >0
    acc.check_pint_quantity(
        tube_inner_diameter,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        blockage_diameter,
        'length',
        ensure_positive=True
    )

    # make sure units cancel
    blockage_diameter = blockage_diameter.to_base_units()
    tube_inner_diameter = tube_inner_diameter.to_base_units()

    # ensure blockage diameter < tube diameter
    if blockage_diameter >= tube_inner_diameter:
        raise ValueError('blockage diameter >= tube diameter')

    # calculate blockage ratio
    blockage_ratio = (1 - (1 - 2 * blockage_diameter.magnitude /
                           tube_inner_diameter.magnitude)**2) * 100

    return blockage_ratio


def calculate_window_sf(
        length,
        width,
        thickness,
        pressure,
        rupture_modulus
):
    """
    This function calculates the safety factor of a clamped rectangular window
    given window dimensions, design pressure, and material rupture modulus

    Parameters
    ----------
    length : pint quantity with length units
        Window unsupported (viewing) length
    width : pint quantity with length units
        Window unsupported (viewing) width
    thickness : pint quantity with length units
        Window thickness
    pressure : pint quantity with pressure units
        Design pressure differential across window at which factor of safety is
        to be calculated
    rupture_modulus : pint quantity with pressure units
        Rupture modulus of desired window material.

    Returns
    -------
    safety_factor : float
        Window factor of safety
    """

    acc.check_pint_quantity(
        length,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        width,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        thickness,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        pressure,
        'pressure',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        rupture_modulus,
        'pressure',
        ensure_positive=True
    )

    safety_factor = acc.window_sympy_solver(
        length=length.to_base_units().magnitude,
        width=width.to_base_units().magnitude,
        thickness=thickness.to_base_units().magnitude,
        pressure=pressure.to_base_units().magnitude,
        rupture_modulus=rupture_modulus.to_base_units().magnitude
    )

    return safety_factor


def calculate_window_thk(
        length,
        width,
        safety_factor,
        pressure,
        rupture_modulus
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
        Design pressure differential across window at which factor of safety is
        to be calculated
    rupture_modulus : pint quantity with pressure units
        Rupture modulus of desired window material.

    Returns
    -------
    thickness : pint quantity
        Window thickness
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    acc.check_pint_quantity(
        length,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        width,
        'length',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        pressure,
        'pressure',
        ensure_positive=True
    )
    acc.check_pint_quantity(
        rupture_modulus,
        'pressure',
        ensure_positive=True
    )

    # Ensure safety factor is numeric and > 1
    try:
        if safety_factor < 1:
            raise ValueError('Window safety factor < 1')
    except TypeError:
        raise TypeError('Non-numeric window safety factor')

    thickness = acc.window_sympy_solver(
        length=length.to_base_units().magnitude,
        width=width.to_base_units().magnitude,
        safety_factor=safety_factor,
        pressure=pressure.to_base_units().magnitude,
        rupture_modulus=rupture_modulus.to_base_units().magnitude
    )

    return quant(thickness, width.to_base_units().units)


def get_pipe_dlf(
        pipe_material,
        pipe_schedule,
        nominal_pipe_size,
        cj_speed,
        plus_or_minus=0.1
):
    """
    This function calculates the dynamic load factor by which a detonation
    tube's static analysis should be scaled in order to account for the tube's
    response to pressure transients. DLF is based on the work of Shepherd [1].
    Since the limits of "approximately equal to" are not define we assume a
    default value of plus or minus ten percent, thus plus_or_minus=0.1.

    [1] Shepherd, J. E. (2009). Structural Response of Piping to Internal Gas
    Detonation. Journal of Pressure Vessel Technology, 131(3), 031204.
    https://doi.org/10.1115/1.3089497

    Parameters
    ----------
    pipe_material : str
        Material which the pipe is made of, e.g. '316L', '304'
    pipe_schedule: str
        The pipe's schedule, e.g. '40', '80s', 'XXS'
    nominal_pipe_size : str
        Nominal pipe size of the detonation tube, e.g. '1/2', '1 1/4', '20'
    cj_speed : pint quantity
        A pint quantity with velocity units representing the Chapman-Jouguet
        wave speed of the detonation in question
    plus_or_minus : float
        Defines the band about the critical velocity which is considered
        "approximately equal to" -- the default value of 0.1 means plus or minus
        ten percent.

    Returns
    -------
    dynamic_load_factor : float
        Factor by which the tube's static maximum pressure should be de-rated to
        account for transient response to detonation waves.
    """
    acc.check_pint_quantity(
        cj_speed,
        'velocity',
        ensure_positive=True
    )

    if not (0 < plus_or_minus < 1):
        raise ValueError('plus_or_minus factor outside of (0, 1)')

    # get pipe dimensions
    [pipe_id,
     pipe_od,
     pipe_thk] = acc.get_pipe_dimensions(
        pipe_schedule,
        nominal_pipe_size
    )

    # get material properties
    properties_dataframe = acc.collect_tube_materials().set_index('Grade')
    if pipe_material not in properties_dataframe.index:
        raise ValueError('Pipe material not found in materials_list.csv')
    elastic_modulus = properties_dataframe['ElasticModulus'][pipe_material].\
        to('Pa').magnitude
    density = properties_dataframe['Density'][pipe_material].\
        to('kg/m^3').magnitude
    poisson = properties_dataframe['Poisson'][pipe_material]

    # set geometry
    pipe_thk = pipe_thk.to('m').magnitude
    pipe_od = pipe_od.to('m').magnitude
    pipe_id = pipe_id.to('m').magnitude
    radius = np.average([pipe_od, pipe_id]) / 2.

    # calculate critical velocity
    crit_velocity = (
        (elastic_modulus ** 2 * pipe_thk ** 2) /
        (3. * density ** 2 * radius ** 2 * (1. - poisson ** 2))
    ) ** (1. / 4)

    # set limits for 'approximately Vcrit'
    bounds = crit_velocity * np.array([
        1. + plus_or_minus,
        1. - plus_or_minus
    ])

    cj_speed = cj_speed.to('m/s').magnitude
    if cj_speed < bounds[1]:
        dynamic_load_factor = 1
    elif cj_speed > bounds[0]:
        dynamic_load_factor = 2
    else:
        dynamic_load_factor = 4

    return dynamic_load_factor


def calculate_ddt_run_up(
        blockage_ratio,
        tube_diameter,
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        phase_specification=''
):
    """
    Calculates the runup distance needed for a detonation to develop from a
    deflagration for a given blockage ratio, tube diameter, and mixture. This is
    accomplished using equations collected by Ciccarelli and Dorofeev [1] for
    blockage ratios <= 0.75. If the desired blockage ratio is less than 0.3,
    the mixture viscosity is needed, and the phase_specification option may be
    necessary depending on the mechanism.

    [1] G. Ciccarelli and S. Dorofeev, “Flame acceleration and transition to
    detonation in ducts,” Prog. Energy Combust. Sci., vol. 34, no. 4, pp.
    499–550, Aug. 2008.

    Parameters
    ----------
    blockage_ratio : float
        Ratio of the cross-sectional area of the detonation tube and a periodic
        blockage used to cause DDT
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
    phase_specification : str
        (Optional) Phase specification within the mechanism file used to
        evaluate thermophysical properties. If Gri30.cti is used with no
        phase specification, viscosity calculations will fail, resulting in an
        error for all blockage ratios less than 0.3.

    Returns
    -------
    runup_distance : pint quantity
        Predicted DDT distance, with the same units as the tube diameter
    """

    if blockage_ratio <= 0 or blockage_ratio > 0.75:
        raise ValueError('Blockage ratio outside of correlation range')

    acc.check_pint_quantity(
        tube_diameter,
        'length',
        ensure_positive=True
    )

    acc.check_pint_quantity(
        initial_temperature,
        'temperature',
        ensure_positive=True
    )

    acc.check_pint_quantity(
        initial_pressure,
        'pressure',
        ensure_positive=True
    )

    # create unit registry and convert tube diameter to avoid ureg issues
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity
    tube_diameter = quant(
        tube_diameter.magnitude,
        tube_diameter.units.format_babel()
    )

    # calculate laminar flamespeed
    laminar_fs = acc.calculate_laminar_flamespeed(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism
    )
    laminar_fs = quant(
        laminar_fs.magnitude, laminar_fs.units.format_babel()
    )

    # calculate density ratio across the deflagration assuming adiabatic flame
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
    sound_speed = acc.get_equil_sound_speed(
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
        Calculate runup distance for blockage ratios <= 0.1 using equation 4.1
        from G. Ciccarelli and S. Dorofeev, “Flame acceleration and transition
        to detonation in ducts,” Prog. Energy Combust. Sci., vol. 34, no. 4, pp.
        499–550, Aug. 2008.
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
            (
                eta *
                (density_ratio - 1)**2 *
                laminar_fs
            ) *
            (
                delta /
                tube_diameter
            )**(1./3)
        )**(1 / (2 * mm + 7. / 3))

        # calculate runup distance
        d_over_h = (
            2. /
            (1 - np.sqrt(1 - blockage_ratio))
        )
        runup = (
                (
                    gamma / cc
                ) * (
                    1 / kappa * np.log(gamma * d_over_h) + kk
                ) * tube_diameter
        )
        return runup.to(tube_diameter.units.format_babel())

    def eq4_4():
        """
        Calculate runup for blockage ratios between 0.3 and 0.75 using equation
        4.4 in G. Ciccarelli and S. Dorofeev, “Flame acceleration and transition
        to detonation in ducts,” Prog. Energy Combust. Sci., vol. 34, no. 4,
        pp. 499–550, Aug. 2008.
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

# TODO: fix bolt calcs
# def calc_single_bolt_stress_areas(
#         bolt_size,
#         bolt_class,
#         bolt_max_stress
# ):
#     ureg = pint.UnitRegistry()
#     quant = ureg.Quantity
#
#     acc.check_pint_quantity(
#         bolt_max_stress,
#         'pressure',
#         ensure_positive=True
#     )
#
#     bolt = (
#     plate = ()
#
#     # get bolt thread specs
#     specs = acc.import_thread_specs()
#     tpi = acc.get_thread_tpi(bolt_size)
#
#     if bolt_max_stress.to('psi').magnitude < 100000:
#         # http://www.engineersedge.com/thread_stress_area_a.htm < 100 ksi
#         diameter = quant(
#             acc.get_thread_property(
#                 'basic diameter',
#                 bolt_size,
#                 bolt_class,
#                 specs['external']
#             ).to('in').magnitude, 'in'
#         )
#         bolt['tensile'] = (
#                 np.pi / 4 * (diameter - quant(0.9743, 'inch') / tpi)**2
#         )
#
#     else:
#         # http://www.engineersedge.com/thread_stress_area_b.htm > 100 ksi
#         diameter = quant(
#             acc.get_thread_property(
#                 'pitch diameter min',
#                 bolt_size,
#                 bolt_class,
#                 specs['external']
#             ).to('in').magnitude, 'in'
#         )
#         bolt['tensile'] = (
#                 np.pi * (diameter / 2 - quant(0.16238, 'inch') / tpi)**2
#         )
#
#
#     return [bolt, plate]
