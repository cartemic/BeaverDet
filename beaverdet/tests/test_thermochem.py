# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for experiments.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import pint
import pytest
import numpy as np
from .. import thermochem

ureg = pint.UnitRegistry()
quant = ureg.Quantity


def test_calculate_laminar_flamespeed():
    """
    Tests the calculate_laminar_flamespeed function
    """

    initial_temperature = quant(300, 'K')
    initial_pressure = quant(1, 'atm')

    # test with bad species
    species = {
        'Wayne': 3,
        'CH4': 7,
        'Garth': 5
    }
    bad_string = 'Species not in mechanism:\nWayne\nGarth\n'
    with pytest.raises(ValueError, match=bad_string):
        thermochem.calculate_laminar_flamespeed(
            initial_temperature,
            initial_pressure,
            species,
            'gri30.cti'
        )

    # test with no species
    species = {}
    with pytest.raises(ValueError, match='Empty species dictionary'):
        thermochem.calculate_laminar_flamespeed(
            initial_temperature,
            initial_pressure,
            species,
            'gri30.cti'
        )

    # test with good species
    species = {
        'CH4': 0.095057034220532327,
        'O2': 0.19011406844106465,
        'N2': 0.71482889733840305
    }
    good_result = 0.39  # value approximated from Law fig. 7.7.7
    test_flamespeed = thermochem.calculate_laminar_flamespeed(
        initial_temperature,
        initial_pressure,
        species,
        'gri30.cti'
    )
    assert abs(test_flamespeed.magnitude - good_result) / good_result < 0.05


def test_eq_sound_speed():
    """
    Tests get_eq_sound_speed
    """

    # check air at 1 atm and 20°C against ideal gas calculation
    gamma = 1.4
    rr = 8.31451
    tt = 293.15
    mm = 0.0289645
    c_ideal = np.sqrt(gamma*rr*tt/mm)

    temp = quant(20, 'degC')
    press = quant(1, 'atm')
    species = {'O2': 1, 'N2': 3.76}
    mechanism = 'gri30.cti'
    c_test = thermochem.get_eq_sound_speed(
        temp,
        press,
        species,
        mechanism
    )

    assert abs(c_ideal - c_test.to('m/s').magnitude) / c_ideal <= 0.005
