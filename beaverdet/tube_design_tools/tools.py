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
import warnings
from math import sqrt
import pint
import pandas as pd
from . import accessories as acc


def get_flange_limits_from_csv(group=2.3):
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
    materials_dict = acc.collect_tube_materials()

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


def calculate_spiral_diameter(pipe_id, blockage_ratio):
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


def get_blockage_ratio(tube_inner_diameter, blockage_diameter):
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

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()

    # ensure tube diameter is a positive numeric pint quantity with a length
    # scale
    try:
        tube_inner_diameter = tube_inner_diameter.to(ureg.inch)
        float(tube_inner_diameter.magnitude)
    except AttributeError:
        # non-pint input
        raise ValueError('tube diameter is not a pint quantity')
    except pint.DimensionalityError:
        # input with bad units
        raise ValueError('tube diameter has bad units')
    except ValueError:
        # non-numeric input
        raise ValueError('tube diameter is non-numeric')
    if tube_inner_diameter.magnitude <= 0:
        raise ValueError('tube diameter <= 0')

    # ensure blockage diameter is a positive numeric pint quantity with a
    # length scale, and also that it is less than the tube diameter
    try:
        blockage_diameter = blockage_diameter.to(ureg.inch)
        float(blockage_diameter.magnitude)
    except AttributeError:
        # non-pint input
        raise ValueError('blockage diameter is not a pint quantity')
    except pint.DimensionalityError:
        # input with bad units
        raise ValueError('blockage diameter has bad units')
    except ValueError:
        # non-numeric input
        raise ValueError('blockage diameter is non-numeric')
    if blockage_diameter.magnitude < 0:
        raise ValueError('blockage diameter < 0')
    elif blockage_diameter >= tube_inner_diameter:
        raise ValueError('blockage diameter >= tube diameter')

    # calculate blockage ratio
    blockage_ratio = (1 - (1 - 2 * blockage_diameter.magnitude /
                           tube_inner_diameter.magnitude)**2) * 100

    return blockage_ratio
