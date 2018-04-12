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
import pandas as pd


def read_flange_csv(group=2.3):
    """
    Reads in flange pressure limits as a function of temperature for different
    pressure classes per ASME B16.5. Temperature is in Centigrade and pressure
    is in bar.

    inputs:
        group (float or string): ANSI B16.5 material group (defaults to 2.3).
                    Only groups 2.2, and 2.3 are included in the current
                    release.

    outputs:
        flange_limits (pandas dataframe): First column is temperature. All
                    other columns' keys are flange classes, and the values
                    are the appropriate pressure limits in bar.
    """

    # ensure group is valid
    group = str(group).replace('.', '_')
    file_name = 'ASME_B16_5_flange_ratings_group_' + group + '.csv'
    if exists(file_name):
        # import the correct .csv file as a pandas dataframe
        flange_limits = pd.read_csv(file_name)
        return flange_limits

    else:
        # the user gave a bad group label
        raise ValueError('{0} is not a valid group'.format(group))


def collect_tube_materials():
    """
    Reads in a csv file containing tube materials and their corresponding
    ANSI B16.5 material groups. This should be used to
        (a) determine available materials and
        (b) determine the correct group so that flange pressure limits can be
            found as a function of temperature

    outputs:
        dictionary with metal names as keys and material groups as values
    """
    file_name = 'materials_list.csv'
    if exists(file_name):
        with open(file_name) as f:
            output_dict = {}
            for num, line in enumerate(f):
                if num > 0:
                    line = line.strip().split(',')
                    output_dict[line[0]] = line[1]
    return output_dict


if __name__ == '__main__':
    # run unit tests
    import test_design_tools
    test_design_tools.unittest.main()
