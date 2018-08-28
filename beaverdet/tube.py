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
from math import sqrt
import pint
import numpy as np
import cantera as ct
import sd2
from . import tools


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
    materials_dict = tools.get_material_groups()

    # ensure desired_material is in materials_dict
    if desired_material not in materials_dict.keys():
        raise ValueError('Desired material not in database.')
    else:
        # material is good, get ASME B16.5 material group
        group = materials_dict[desired_material]

    # initialize unit registry for unit handling
    ureg = pint.UnitRegistry()

    # type check to make sure temperature is a pint quantity
    tools.check_pint_quantity(
        temperature,
        'temperature'
    )

    # type check to make sure pressure is a pint quantity
    tools.check_pint_quantity(
        pressure,
        'pressure'
    )

    # import flange limits from csv
    flange_limits = tools.get_flange_limits_from_csv(group)

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
        blockage_ratio: percentage (float between 0 and 1)

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

    # ensure blockage ratio is on 0<BR<1
    if not 0 < blockage_ratio < 1:
        raise ValueError('Blockage ratio outside of 0<BR<1')

    tools.check_pint_quantity(
        pipe_id,
        'length',
        ensure_positive=True
    )

    # calculate Shchelkin spiral diameter
    spiral_diameter = pipe_id / 2 * (1 - sqrt(1 - blockage_ratio))
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
        blockage_ratio: float between 0 and 1
    """

    # check dimensionality and >0
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
    if blockage_diameter >= tube_inner_diameter:
        raise ValueError('blockage diameter >= tube diameter')

    # calculate blockage ratio
    blockage_ratio = (1 - (1 - 2 * blockage_diameter.magnitude /
                           tube_inner_diameter.magnitude)**2)

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

    safety_factor = tools.window_sympy_solver(
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
            raise ValueError('Window safety factor < 1')
    except TypeError:
        raise TypeError('Non-numeric window safety factor')

    thickness = tools.window_sympy_solver(
        length=length.to_base_units().magnitude,
        width=width.to_base_units().magnitude,
        safety_factor=safety_factor,
        pressure=pressure.to_base_units().magnitude,
        rupture_modulus=rupture_modulus.to_base_units().magnitude
    )

    return quant(
        thickness,
        width.to_base_units().units).to(width.units.format_babel())


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
    tools.check_pint_quantity(
        cj_speed,
        'velocity',
        ensure_positive=True
    )

    if not (0 < plus_or_minus < 1):
        raise ValueError('plus_or_minus factor outside of (0, 1)')

    # get pipe dimensions
    dimensions = tools.get_pipe_dimensions(
        pipe_schedule,
        nominal_pipe_size
    )
    pipe_od = dimensions['outer diameter']
    pipe_id = dimensions['inner diameter']
    pipe_thk = dimensions['wall thickness']

    # get material properties
    properties_dataframe = tools.collect_tube_materials().set_index('Grade')
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

    # create unit registry and convert tube diameter to avoid ureg issues
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity
    tube_diameter = quant(
        tube_diameter.magnitude,
        tube_diameter.units.format_babel()
    )

    # calculate laminar flamespeed
    laminar_fs = tools.calculate_laminar_flamespeed(
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
    sound_speed = tools.get_equil_sound_speed(
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
            (eta * (density_ratio - 1)**2 * laminar_fs) *
            (delta / tube_diameter)**(1./3)
        )**(1 / (2 * mm + 7. / 3))

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


def calculate_bolt_stress_areas(
        thread_size,
        thread_class,
        bolt_max_tensile,
        plate_max_tensile,
        engagement_length
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

    Returns
    -------
    thread : dict
        Dictionary with the following key/value pairs:
        'plate area': stress area of internal threads within the plate
        'screw area': stress area of external threads on the screw
        'minimum engagement': minimum engagement length causing screw to fail
            in tension rather than shear, thus preventing the plate from
            stripping.
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

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
    thread_specs = tools.import_thread_specs()
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
        )**2
    else:
        # calculate screw tensile area using eq. 2b (p. 1490) in Fasteners
        # section of Machinery's Handbook 26
        screw_area_tensile = np.pi * (
            e_s_min / 2 -
            0.16238 / tpi
        )**2

    # calculate screw shear area using eq. 5 (p. 1491) in Fasteners section of
    # Machinery's Handbook 26
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

    # calculate plate shear area using eq. 6 (p. 1491) in Fasteners section of
    # Machinery's Handbook 26
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

    # calculate minimum thread engagement (corrected for material differences)
    # using eqs. 1 and 4 (pp. 1490-1491) in Fasteners section of Machinery's
    # Handbook 26
    thread['minimum engagement'] = (
        2 * screw_area_tensile /
        (k_n_max * np.pi * (
            1. / 2 + 0.57735 * tpi * (e_s_min - k_n_max)
        )
         )
    ) * j_factor

    return thread


def calculate_window_bolt_sf(
        max_pressure,
        window_area,
        num_bolts,
        thread_size,
        thread_class,
        bolt_max_tensile,
        plate_max_tensile,
        engagement_length
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

    Returns
    -------
    safety_factor : dict
        Dictionary with keys of 'bolt' and 'plate', giving factors of safety
        for window bolts and the plate that they are screwed into.
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

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
    window_force = (max_pressure - quant(1, 'atm')) * window_area / num_bolts

    # get stress areas
    thread = calculate_bolt_stress_areas(
        thread_size,
        thread_class,
        bolt_max_tensile,
        plate_max_tensile,
        engagement_length
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
    safety_factor = dict()
    safety_factor['bolt'] = (
        bolt_max_tensile / (window_force / screw_area)
    ).to_base_units()
    safety_factor['plate'] = (
        plate_max_tensile / (window_force / plate_area)
    ).to_base_units()
    return safety_factor


def calculate_reflected_shock_state(
        initial_pressure,
        initial_temperature,
        species_dict,
        mechanism
):
    """
    Calculates the thermodynamic and chemical state of a reflected shock using
    sd2.

    Parameters
    ----------
    initial_pressure : pint quantity
        Pint quantity of mixture initial pressure
    initial_temperature : pint quantity
        Pint quantity of mixture initial temperature
    species_dict : dict
        Dictionary of initial reactant mixture
    mechanism : str
        Mechanism to use for chemical calculations, e.g. 'gri30.cti'

    Returns
    -------
    dict
        Dictionary containing keys 'reflected' and 'cj'. Each of these contains
        'speed', indicating the related wave speed, and 'state', which is a
        Cantera gas object at the specified state.
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    tools.check_pint_quantity(
        initial_pressure,
        'pressure',
        ensure_positive=True
    )

    tools.check_pint_quantity(
        initial_temperature,
        'temperature',
        ensure_positive=True
    )

    # define gas objects
    initial_gas = ct.Solution(mechanism)
    reflected_gas = ct.Solution(mechanism)

    # define gas states
    initial_temperature = initial_temperature.to('K').magnitude
    initial_pressure = initial_pressure.to('Pa').magnitude
    initial_gas.TPX = [
        initial_temperature,
        initial_pressure,
        species_dict
    ]
    reflected_gas.TPX = [
        initial_temperature,
        initial_pressure,
        species_dict
    ]

    # get CJ state
    [cj_speed,
     cj_gas] = sd2.detonations.calculate_cj_speed(
        initial_pressure,
        initial_temperature,
        species_dict,
        mechanism,
        return_state=True
    )

    # get reflected state
    [_,
     reflected_speed,
     reflected_gas] = sd2.shocks.get_reflected_equil_state_0(
        initial_gas,
        cj_gas,
        reflected_gas,
        cj_speed
    )

    return {'reflected': {'speed': quant(reflected_speed, 'm/s'),
                          'state': reflected_gas},
            'cj': {'speed': quant(cj_speed, 'm/s'),
                   'state': cj_gas}
            }


def calculate_max_initial_pressure(
        pipe_material,
        pipe_schedule,
        pipe_nps,
        welded,
        desired_fs,
        initial_temperature,
        species_dict,
        mechanism,
        error_tol=1e-4,
        max_pressure=False,
        max_iterations=500
):
    """
    Iteratively calculates the maximum initial pressure for a given detonation
    tube and reactant mixture.

    Parameters
    ----------
    pipe_material : str
        Material that pipe is made of, e.g. '316L'
    pipe_schedule : str
        Pipe schedule, e.g. '80', 'XXS'
    pipe_nps : str
        Nominal pipe size in inches, e.g. '6' for NPS-6
    welded : bool
        True for welded pipe, False for seamless
    desired_fs : float
        Desired tube factor of safety
    initial_temperature : pint quantity
        Pint quantity of initial mixture temperature
    species_dict : dict
        Dictionary of reactant mixture components
    mechanism : str
        Mechanism to use for calculations, e.g. 'gri30.cti'
    error_tol : float
        Relative error tolerance below which initial pressure calculations are
        considered 'good enough'
    max_pressure : pint quantity
        Pint quantity with maximum total pressure. Defaults to False if nothing
        is specified, meaning that max pressure will be calculated from pipe
        material properties.
    max_iterations : int
        Maximum number of loop iterations before exit, defaults to 500

    Returns
    -------
    initial_pressure : pint quantity
        Pint quantity of max allowable initial pressure
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # ensure temperature is a pint quantity, and convert it to# the local unit
    # registry to avoid problems
    tools.check_pint_quantity(
        initial_temperature,
        'temperature',
        ensure_positive=True
    )
    initial_temperature = quant(
        initial_temperature.magnitude,
        initial_temperature.units.format_babel()
    )

    # define tube dimensions
    dimensions = tools.get_pipe_dimensions(
        pipe_schedule,
        pipe_nps
    )
    tube_od = dimensions['outer diameter']
    tube_id = dimensions['inner diameter']
    wall_thickness = dimensions['wall thickness']
    tube_od = quant(tube_od.magnitude, tube_od.units.format_babel())
    tube_id = quant(tube_id.magnitude, tube_id.units.format_babel())
    wall_thickness = quant(wall_thickness.magnitude,
                           wall_thickness.units.format_babel())

    # look up max allowable stress
    stress_limits = tools.get_pipe_stress_limits(
        pipe_material,
        welded
    )
    temp_units = stress_limits['temperature'][0]
    temperatures = stress_limits['temperature'][1]
    # ensure material stress limits have monotonically increasing temperatures,
    # otherwise the np.interp "results are nonsense" per scipy docs
    if not np.all(np.diff(temperatures) > 0):
        raise ValueError('Stress limits require temperatures to be ' +
                         'monotonically increasing')
    stress_units = stress_limits['stress'][0]
    stresses = stress_limits['stress'][1]
    current_temp = initial_temperature.to(temp_units).magnitude
    max_stress = quant(np.interp(current_temp, temperatures, stresses),
                       stress_units)

    # calculate max allowable pressure
    if not max_pressure:
        # user didn't give max pressure; calculate it using basic longitudinal
        # joint formula on page 14 of Megyesy's Pressure Vessel Handbook, 8th
        # ed.
        mean_diameter = (tube_od + tube_id) / 2
        asme_fs = 4
        max_allowable_pressure = (
                max_stress * (2 * wall_thickness) * asme_fs /
                (mean_diameter * desired_fs)
        )
    else:
        # make sure it's a pint quantity with pressure units and use it
        tools.check_pint_quantity(
            max_pressure,
            'pressure',
            ensure_positive=True
        )
        max_allowable_pressure = quant(
            max_pressure.magnitude,
            max_pressure.units.format_babel()
        )

    # define error and pressure initial guesses and start loop
    initial_pressure = quant(1, 'atm')
    error = 1000
    counter = 0
    while error > error_tol and counter < max_iterations:
        counter += 1
        # get reflected shock pressure
        states = calculate_reflected_shock_state(
                initial_pressure,
                initial_temperature,
                species_dict,
                mechanism
            )

        reflected_pressure = states['reflected']['state'].P
        reflected_pressure = quant(
            reflected_pressure,
            'Pa'
        )
        cj_speed = states['cj']['speed']
        cj_speed = quant(cj_speed.to('m/s').magnitude, 'm/s')

        # get dynamic load factor
        dlf = get_pipe_dlf(
            pipe_material,
            pipe_schedule,
            pipe_nps,
            cj_speed
        )

        # calculate error, accounting for dynamic load factor
        error = abs(
            reflected_pressure.to_base_units().magnitude -
            max_allowable_pressure.to_base_units().magnitude / dlf) / \
            (max_allowable_pressure.to_base_units().magnitude / dlf)

        # find new initial pressure
        initial_pressure = (
                initial_pressure *
                max_allowable_pressure.to_base_units().magnitude /
                dlf /
                reflected_pressure.to_base_units().magnitude
        )

    return initial_pressure

# TODO: Tie everything together
