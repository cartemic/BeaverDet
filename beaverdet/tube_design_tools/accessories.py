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


def check_materials():
    """
    Makes sure that the materials in materials_list.csv have stress limits and
    flange ratings. This function relies on collect_tube_materials().
    """
    # collect files
    file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'lookup_data')
    my_files = os.listdir(file_directory)
    flange_ratings = [file for file in my_files if "flange" in file.lower()]
    stress_limits = [file for file in my_files if "stress" in file.lower()]
    materials_list = collect_tube_materials()

    # make sure things were actually loaded
    if not bool(flange_ratings + stress_limits):
        raise ValueError('no files containing "flange" or "stress" found')

    # make sure all pipe material limits are either welded or seamless
    # other types are permitted, but will raise a warning
    for file in stress_limits:
        if ('welded' not in file.lower()) and ('seamless' not in file.lower()):
            # warn that something is weird
            warnings.warn(file +
                          'does not indicate whether it is welded or seamless')

        # initialize an error string and error indicator. Error string will be
        # used to aggregate errors in the list of available materials so that
        # all issues may be rectified simultaneously.
        error_string = '\n'
        has_errors = False

        # check the first row of the file in question to extract the names of
        # the materials that it contains stress limits for
        with open(file_directory + file, 'r') as current_file:
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
    Reads in a csv file containing tube materials and their corresponding
    ASME B16.5 material groups. This should be used to
        a. determine available materials and
        b. determine the correct group so that flange pressure limits can be
            found as a function of temperature

    Inputs:
        none

    Outputs:
       tube_materials:  dictionary with metal names as keys and
       material groups as values
    """
    file_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'lookup_data')
    file_name = 'materials_list.csv'
    file_location = os.path.relpath(os.path.join(file_directory, file_name))
    output_dict = {}

    # read in csv and extract information
    if os.path.exists(file_location):
        with open(file_location) as file:
            for num, line in enumerate(file):
                # skip the first line of the file, since it contains only
                # column titles and no data
                if num > 0:
                    # extract each material name and corresponding group
                    line = line.strip().split(',')
                    output_dict[line[0]] = line[1]

                    # warn the user if materials_list.csv has more than 2
                    # columns
                    if len(line) > 2:
                        warnings.warn(file_name + ' contains extra entries')
    else:
        # raise an exception if the file doesn't exist
        raise ValueError(file_name + ' does not exist')

    # raise an exception if the file is empty
    if not bool(output_dict):
        raise ValueError(file_name + ' is empty')

    return output_dict


def check_pint_quantity(quantity, dimension_type, ensure_positive=False):
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
        if quantity.magnitude < 0:
            raise ValueError('Input value < 0')

    if units[dimension_type] != actual_dimension_type:
        raise ValueError(
            actual_dimension_type +
            ' is not '
            + units[dimension_type]
        )

    return True


def window_sympy_solver(
        **kwargs):
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
