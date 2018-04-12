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
import pint
import pandas as pd


def read_flange_csv(group=2.3):
    """
    Reads in flange pressure limits as a function of temperature for different
    pressure classes per ASME B16.5. Temperature is in Centigrade and pressure
    is in bar.

    Inputs:
        group (float or string): ANSI B16.5 material group (defaults to 2.3).
                    Only groups 2.2, and 2.3 are included in the current
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
    Q_ = ureg.Quantity
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
        flange_limits['Temperature'] = [Q_(temp, ureg.degC) for temp in
                                        flange_limits['Temperature']]

        # add units to pressure columns
        for key in flange_limits.keys():
            if key != 'Temperature':
                flange_limits[key] = [Q_(pressure, ureg.bar) for pressure in
                                      flange_limits[key]]

        return flange_limits

    else:
        # the user gave a bad group label
        raise ValueError('{0} is not a valid group'.format(group))


def collect_tube_materials():
    """
    Reads in a csv file containing tube materials and their corresponding
    ANSI B16.5 material groups. This should be used to
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


if __name__ == '__main__':
    read_flange_csv()
