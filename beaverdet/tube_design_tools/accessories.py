# -*- coding: utf-8 -*-
"""
PURPOSE:
    A series of accessories used in the function of the detonation design
    tools found in tools.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import os
import warnings
import pint
import sympy as sp
import numpy as np
import pandas as pd
import cantera as ct


def check_materials():
    """
    Makes sure that the materials in materials_list.csv have stress limits and
    flange ratings. This function relies on get_material_groups().
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
    materials_list = get_material_groups()

    # make sure things were actually loaded
    if not bool(flange_ratings + stress_limits):
        raise ValueError('no files containing "flange" or "stress" found')

    # initialize an error string and error indicator. Error string will be
    # used to aggregate errors in the list of available materials so that
    # all issues may be rectified simultaneously.
    error_string = '\n'
    has_errors = False

    # make sure all pipe material limits are either welded or seamless
    # other types are permitted, but will raise a warning
    for file in stress_limits:
        if ('welded' not in file.lower()) and ('seamless' not in file.lower()):
            # warn that something is weird
            warnings.warn(file +
                          'does not indicate whether it is welded or seamless')

        # check the first row of the file in question to extract the names of
        # the materials that it contains stress limits for
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
            for item in materials_list:
                if item not in materials:
                    # a material is missing from the limits spreadsheet.
                    # indicate that an error has occurred, and add it to the
                    # error string.
                    error_string += 'Material ' + item + ' not found in ' +\
                                    file + '\n'
                    has_errors = True

    # find out which material groups need to be inspected
    groups = set()
    for _, group in materials_list.items():
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


def collect_tube_materials():
    """
    Reads in a csv file containing tube materials, their corresponding
    ASME B16.5 material groups, and selected material properties.

    Returns
    -------
    materials_dataframe : pandas dataframe
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
            raise ValueError(file_name + ' is empty')

    else:
        # raise an exception if the file doesn't exist
        raise ValueError(file_name + ' does not exist')

    # apply units
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity
    materials_dataframe.ElasticModulus = [
        quant(item, 'GPa') for item in materials_dataframe.ElasticModulus.values
    ]
    materials_dataframe.Density = [
        quant(item, 'g/cm^3') for item in materials_dataframe.Density.values
    ]

    return materials_dataframe


def get_material_groups():
    """
    Collects materials and their associated ASME B16.5 material groups from a
    dataframe of material properties

    Returns
    -------
    groups_dict
    """
    materials_dataframe = collect_tube_materials()
    grades = materials_dataframe.Grade.values.astype(str)
    groups = materials_dataframe.Group.values.astype(str)
    groups_dict = {}
    for [grade, group] in zip(grades, groups):
        groups_dict[grade] = group

    return groups_dict


def check_pint_quantity(
        quantity,
        dimension_type,
        ensure_positive=False
):
    """
    This function checks to make sure that a quantity is an instance of a pint
    quantity, and that it has the correct units.

    Currently supported dimension types:
        length
        pressure
        temperature
        velocity

    Parameters
    ----------
    quantity : pint quantity
        Pint quantity which is to be checked for dimensionality
    dimension_type : str
        Dimensionality that quantity should have
    ensure_positive : bool
        Determines whether the magnitude of the pint quantity will be checked
        for positivity

    Returns
    -------
    True if no errors are raised

    """

    ureg = pint.UnitRegistry()
    units = {
        'length': ureg.meter.dimensionality.__str__(),
        'temperature': ureg.degC.dimensionality.__str__(),
        'pressure': ureg.psi.dimensionality.__str__(),
        'velocity': (ureg.meter/ureg.second).dimensionality.__str__()
    }

    if dimension_type not in units:
        raise ValueError(dimension_type + ' not a supported dimension type')

    try:
        actual_dimension_type = quantity.dimensionality.__str__()
    except AttributeError:
        raise ValueError('Non-pint quantity')

    try:
        float(quantity.magnitude)
    except ValueError:
        raise ValueError('Non-numeric pint quantity')

    if ensure_positive:
        if quantity.to_base_units().magnitude < 0:
            raise ValueError('Input value < 0')

    if units[dimension_type] != actual_dimension_type:
        raise ValueError(
            actual_dimension_type +
            ' is not '
            + units[dimension_type]
        )

    return True


def window_sympy_solver(
        **kwargs
):
    """
    This function uses sympy to solve for a missing window measurement. Inputs
    are five keyword arguments, with the following possible values:
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
    https://www.crystran.co.uk/userfiles/files/design-of-pressure-windows.pdf

    Parameters
    ----------
    kwargs

    Returns
    -------
    missing value as a float, or NaN if the result is imaginary
    """

    # Ensure that 5 keyword arguments were given
    if kwargs.__len__() != 5:
        raise ValueError('Incorrect number of arguments sent to solver')

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
        error_string = 'Bad keyword argument:'
        for arg in bad_args:
            error_string += '\n'+arg

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


def calculate_laminar_flamespeed(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        phase_specifation=''
):
    """
    This function uses cantera to calculate the laminar flame speed of a given
    gas mixture.

    Parameters
    ----------
    initial_temperature : pint quantity
        Initial temperature of gas mixture
    initial_pressure : pint quantity
        Initial pressure of gas mixture
    species_dict : dict
        Dictionary with species names (all caps) as keys and moles as values
    mechanism : str
        String of mechanism to use (e.g. 'gri30.cti')
    phase_specifation : str
        Phase specification for cantera solution

    Returns
    -------
    Laminar flame speed in m/s as a pint quantity
    """
    gas = ct.Solution(mechanism, phase_specifation)

    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    check_pint_quantity(
        initial_pressure,
        'pressure',
        ensure_positive=True
    )
    check_pint_quantity(
        initial_temperature,
        'temperature',
        ensure_positive=True
    )

    # ensure species dict isn't empty
    if len(species_dict) == 0:
        raise ValueError('Empty species dictionary')

    # ensure all species are in the mechanism file
    bad_species = ''
    good_species = gas.species_names
    for species in species_dict:
        if species not in good_species:
            bad_species += species + '\n'
    if len(bad_species) > 0:
        raise ValueError('Species not in mechanism:\n' + bad_species)

    gas.TPX = (
        initial_temperature.to('K').magnitude,
        initial_pressure.to('Pa').magnitude,
        species_dict
    )

    # find laminar flame speed
    flame = ct.FreeFlame(gas)
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    flame.solve(loglevel=0)

    return quant(flame.u[0], 'm/s')


def import_pipe_schedules():
    """
    This function imports pipe schedule dimensions from a fixed location

    Returns
    -------
    schedule_info : pandas dataframe
        Dataframe of available pipe schedules and their associated dimensions
    """
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

    schedule_info = pd.read_csv(file_location, index_col=0)
    return schedule_info


def get_available_pipe_sizes(
        pipe_schedule,
        schedule_info
):
    """
    This function finds available nominal pipe sizes for a given pipe schedule
    (in inches)

    Parameters
    ----------
    pipe_schedule: str
        Desired pipe schedule (e.g. 40, 80s, XXS, etc.)
    schedule_info: pandas dataframe
        Dataframe of available schedules from import_pipe_schedules()

    Returns
    -------
    available_sizes : list
        List of available sizes for the given pipe schedule (in inches)
    """
    try:
        available_sizes = list(
            schedule_info[pipe_schedule].dropna().to_dict().keys()
        )
    except KeyError:
        raise ValueError('Pipe class not found')

    return available_sizes


def get_pipe_dimensions(
        pipe_schedule,
        nominal_size
):
    """
    This function finds the inner and outer diameters and wall thickness of a
    pipe of a given schedule and nominal size.

    Parameters
    ----------
    pipe_schedule : str
        String of desired pipe schedule (e.g. 40, 80s, XXS, etc.)
    nominal_size : str
        String of desired nominal pipe size in inches (e.g. 1, 1/2, 2 1/2, etc.)

    Returns
    -------
    list
        [outer diameter, inner diameter, wall thickness] as pint quantities

    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # enforce string data types
    pipe_schedule = str(pipe_schedule)
    nominal_size = str(nominal_size)

    # collect pipe schedules
    pipe_schedule_dataframe = import_pipe_schedules()
    available_pipe_sizes = get_available_pipe_sizes(
        pipe_schedule,
        pipe_schedule_dataframe
    )

    # ensure size exists
    if nominal_size not in available_pipe_sizes:
        raise ValueError('Nominal size not found for given pipe schedule')

    outer_diameter = pipe_schedule_dataframe['OD'][nominal_size]
    wall_thickness = pipe_schedule_dataframe[pipe_schedule][nominal_size]
    inner_diameter = outer_diameter - 2 * wall_thickness

    return [quant(outer_diameter, ureg.inch),
            quant(inner_diameter, ureg.inch),
            quant(wall_thickness, ureg.inch)]


def import_thread_specs():
    """
    Imports thread specifications from .csv files

    Returns
    -------
    thread_specs : list
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


def get_thread_property(
        thread_property,
        thread_size,
        thread_class,
        thread_specs
):
    """
    Finds a thread property, such as minor diameter, using a dataframe from
    import_thread_specs(). import_thread_specs is not directly called here to
    save time by not reading from disk every time a property is requested.

    Parameters
    ----------
    thread_property : str
        Property that is desired, such as 'minor diameter'
    thread_size : str
        Thread size for desired property, such as '1/4-20' or '1 1/2-6'
    thread_class : str
        Thread class: '2B' or '3B' for internal threads, '2A' or '3A' for
        external threads
    thread_specs : pandas.core.frame.DataFrame
        Pandas dataframe of thread properties, from import_thread_specs()

    Returns
    -------
    pint.UnitRegistry().Quantity
        Property requested, as a pint quantity with units of inches
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # ensure thread_specs is a pandas dataframe
    if not isinstance(thread_specs, pd.DataFrame):
        raise TypeError('thread_specs is not a pandas dataframe')

    # ensure property is a string and in the specs dataframe
    if not isinstance(thread_property, str):
        raise TypeError('thread_property expected a string')
    elif thread_property not in thread_specs.keys():
        raise KeyError('Thread property \'' +
                       thread_property +
                       '\' not found. Available specs: ' +
                       "'" + "', '".join(thread_specs.keys()) + "'")

    # ensure thread size is a string and in the specs dataframe
    if not isinstance(thread_size, str):
        raise TypeError('thread_size expected a string')
    elif thread_size not in thread_specs.index:
        raise KeyError('Thread size \'' +
                       thread_size +
                       '\' not found')

    # ensure thread class is a string and in the specs dataframe
    if not isinstance(thread_class, str):
        raise TypeError('thread_class expected a string')
    elif not any(pd.MultiIndex.isin(thread_specs.index, [thread_class], 1)):
        raise KeyError('Thread class \'' +
                       thread_class +
                       '\' not found')

    # retrieve the property
    return quant(thread_specs[thread_property][thread_size][thread_class], 'in')


def get_thread_tpi(
        thread_size
):
    """
    Gets the number of threads per inch from a string of the thread size, such
    as '1/4-20' -> 20

    Parameters
    ----------
    thread_size : str
        String of the thread size, such as '1/4-20'

    Returns
    -------
    int
        Integer number of threads per inch
    """
    return int(thread_size.split('-')[-1])


def get_equil_sound_speed(
        temperature,
        pressure,
        species_dict,
        mechanism,
        phase_specification=''
):
    """
    Calculates the equilibrium speed of sound in a mixture

    Parameters
    ----------
    temperature : pint quantity
        Initial mixture temperature
    pressure : pint quantity
        Initial mixture pressure
    species_dict : dict
        Dictionary of mixture mole fractions
    mechanism : str
        Desired chemical mechanism
    phase_specification : str
        Phase specification for cantera solution

    Returns
    -------
    sound_speed : pint quantity
        local speed of sound in m/s
    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    check_pint_quantity(
        pressure,
        'pressure',
        ensure_positive=True
    )

    check_pint_quantity(
        temperature,
        'temperature',
        ensure_positive=True
    )

    working_gas = ct.Solution(mechanism, phase_specification)
    working_gas.TPX = [
        temperature.to('K').magnitude,
        pressure.to('Pa').magnitude,
        species_dict
        ]

    pressures = np.zeros(2)
    densities = np.zeros(2)

    # equilibrate gas at input conditions and collect pressure, density
    working_gas.equilibrate('TP')
    pressures[0] = working_gas.P
    densities[0] = working_gas.density

    # perturb pressure and equilibrate with constant P, s to get dp/drho|s
    pressures[1] = 1.0001 * pressures[0]
    working_gas.SP = working_gas.s, pressures[1]
    working_gas.equilibrate('SP')
    densities[1] = working_gas.density

    # calculate sound speed
    sound_speed = np.sqrt(np.diff(pressures)/np.diff(densities))[0]

    return quant(sound_speed, 'm/s')


def get_pipe_stress_limits(
        material,
        welded=False
):
    check_materials()

    # collect files
    file_directory = os.path.join(
        os.path.dirname(
            os.path.relpath(__file__)
        ),
        'lookup_data'
    )
    file_name = 'ASME_B31_1_stress_limits_'
    if welded:
        file_name += 'welded.csv'
    else:
        file_name += 'seamless.csv'
    file_location = os.path.join(
        file_directory,
        file_name
    )
    material_limits = pd.read_csv(file_location, index_col=0)

    if material not in material_limits.keys():
        raise KeyError('material not found')

    material_limits = material_limits[material]

    # apply units
    limits = {
        'temperature': ('degF', []),
        'stress': ('ksi', [])
    }
    for temp, stress in material_limits.items():
        limits['temperature'][1].append(temp)
        limits['stress'][1].append(stress)

    return limits
