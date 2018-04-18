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


from os.path import exists
from os import listdir
import warnings
from math import sqrt
import pint
import pandas as pd


def read_flange_csv(group=2.3):
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
    file_directory = './lookup_data/'
    file_name = 'ASME_B16_5_flange_ratings_group_' + group + '.csv'
    file_location = file_directory + file_name

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity
    if exists(file_location):
        # import the correct .csv file as a pandas dataframe
        flange_limits = pd.read_csv(file_location)

        # ensure all temperatures and pressures are floats, and check to make
        # sure pressures are greater than zero
        values = flange_limits.values
        for row in values:
            for element, item in enumerate(row):
                # ensure each item is a float and assign non-numeric values
                # a value of zero
                try:
                    item = float(item)
                except ValueError:
                    item = 0.

                if element > 0:
                    # these are pressures, which must be positive
                    if item < 0:
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
    file_directory = './lookup_data/'
    file_name = 'materials_list.csv'
    file_location = file_directory + file_name
    if exists(file_location):
        with open(file_location) as file:
            output_dict = {}
            for num, line in enumerate(file):
                if num > 0:
                    line = line.strip().split(',')
                    output_dict[line[0]] = line[1]
    return output_dict


def get_flange_class(temperature, pressure, desired_material):
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
    materials_dict = collect_tube_materials()

    # ensure desired_material is in materials_dict
    if desired_material not in materials_dict.keys():
        raise ValueError('Desired material not in database.')
    else:
        # material is good, get ASME B16.5 material group
        group = materials_dict[desired_material]

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # type check to make sure temperature is a pint quantity
    try:
        temperature.to_base_units()
    except AttributeError:
        # temperature is not a pint quantity. Try to make it one, and if that
        # fails, raise an error
        try:
            # ensure numeric type and convert to quantity
            float(temperature)
            temperature = quant(temperature, ureg.degC)
            # let the user know something went wrong
            warnings.warn('No temperature units. Assuming °C.')
        except ValueError:
            # temperature is non-numeric
            raise ValueError('Non-numeric temperature input.')

    # type check to make sure pressure is a pint quantity
    try:
        pressure.to_base_units()
    except AttributeError:
        # pressure is not a pint quantity. Try to make it one, and if that
        # fails, raise an error
        try:
            # ensure numeric type and convert to quantity
            float(pressure)
            pressure = quant(pressure, ureg.bar)
            # let the user know something went wrong
            warnings.warn('No pressure units. Assuming bar.')
        except ValueError:
            # pressure is non-numeric
            raise ValueError('Non-numeric pressure input.')

    # ensure units are good:
    #   convert temperature to degC
    #   convert pressure to bar
    # return ValueError if this is not possible
    try:
        # check and convert temperature units
        temperature = temperature.to(ureg.degC)
    except pint.DimensionalityError:
        raise ValueError('Bad temperature units.')
    try:
        # check and convert pressure units
        pressure = pressure.to(ureg.bar)
    except pint.DimensionalityError:
        raise ValueError('Bad pressure units.')

    # import flange limits from csv
    flange_limits = read_flange_csv(group)

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


def check_materials():
    """
    Makes sure that the materials in materials_list.csv have stress limits and
    flange ratings. This function relies on collect_tube_materials().
    """
    # collect files
    directory = './lookup_data/'
    my_files = listdir(directory)
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
            warnings.warn(directory + file +
                          'does not indicate whether it is welded or seamless')

        # initialize an error string and error indicator. Error string will be
        # used to aggregate errors in the list of available materials so that
        # all issues may be rectified simultaneously.
        error_string = '\n'
        has_errors = False

        # check the first row of the file in question to extract the names of
        # the materials that it contains stress limits for
        with open(directory + file, 'r') as current_file:
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
                                    directory + file + '\n'
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


def get_spiral_diameter(pipe_id, blockage_ratio):
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

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # ensure blockage ratio is a float
    try:
        blockage_ratio = float(blockage_ratio)
    except ValueError:
        raise ValueError('Non-numeric blockage ratio.')

    # ensure blockage ratio is on 0<BR<100
    if not 0 < blockage_ratio < 100:
        raise ValueError('Blockage ratio outside of 0<BR<100')

    # check inner diameter units to make sure they are length-scale
    try:
        pipe_id.to(ureg.inch)
        # make sure pipe_id is numeric
        try:
            float(pipe_id.magnitude)
        except ValueError:
            # pipe_id is non-numeric quantity
            raise ValueError('ID is non-numeric quantity.')
    except pint.DimensionalityError:
        # diameter has bad units
        raise ValueError('Bad diameter units.')
    except AttributeError:
        # pipe_id is not a pint quantity. check if it is numeric.
        try:
            float(pipe_id)
            # if no error, raise a warning and assume inches
            pipe_id = quant(pipe_id, ureg.inch)
            warnings.warn('No ID units, assuming inches.')
        except ValueError:
            # pipe_id is non-numeric, raise error
            raise ValueError('ID is unitless and non-numeric.')

    # calculate Shchelkin spiral diameter
    spiral_diameter = pipe_id / 2 * (1 - sqrt(1 - blockage_ratio / 100))
    return spiral_diameter


if __name__ == '__main__':
    read_flange_csv()
