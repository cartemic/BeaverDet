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
import warnings
import pint
import pandas as pd


def read_flange_csv(group=2.3):
    """
    Reads in flange pressure limits as a function of temperature for different
    pressure classes per ASME B16.5. Temperature is in Centigrade and pressure
    is in bar.

    Inputs:
        group (float or string): ASME B16.5 material group (defaults to 2.3).
                    Only groups 2.1, 2.2, and 2.3 are included in the current
                    release.

    Outputs:
        flange_limits (pandas dataframe): First column is temperature. All
                    other columns' keys are flange classes, and the values
                    are the appropriate pressure limits in bar.
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
       tube_materials (dictionary):  dictionary with metal names as keys and
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


if __name__ == '__main__':
    read_flange_csv()
