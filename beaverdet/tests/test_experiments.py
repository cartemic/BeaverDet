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

import pytest
import pint
from .. import experiments

ureg = pint.UnitRegistry()
quant = ureg.Quantity


def test_test_matrix():
    good_initial_pressure = quant(1, 'atm')
    good_initial_temperature = quant(20, 'degC')
    good_fuel = 'H2'
    good_oxidizer = 'O2'
    good_diluent = 'AR'
    good_equivalence = [1]
    good_diluent_mole_fraction = [0]
    good_num_replicates = 3
    good_tube_volume = quant(0.1028, 'm^3')

    # <editor-fold desc="__init__()">
    # num_replicates <= 0
    with pytest.raises(
        ValueError,
        match='bad number of replicates'
    ):
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_fuel,
            good_oxidizer,
            good_diluent,
            good_equivalence,
            good_diluent_mole_fraction,
            0,
            good_tube_volume
        )

    # non-iterable numeric equivalence
    matrix = experiments.TestMatrix(
        good_initial_pressure,
        good_initial_temperature,
        good_fuel,
        good_oxidizer,
        good_diluent,
        1,
        good_diluent_mole_fraction,
        good_num_replicates,
        good_tube_volume
    )

    # check to make sure that matrix.equivalence is iterable
    assert len(matrix.equivalence) == 1

    # iterable equivalence with non-numeric values
    with pytest.raises(
        TypeError,
        match='equivalence has non-numeric items'
    ):
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_fuel,
            good_oxidizer,
            good_diluent,
            'iterable and non numeric',
            good_diluent_mole_fraction,
            good_num_replicates,
            good_tube_volume
        )

    # equivalence <= 0
    for bad_equivalence in [[0], [-1]]:
        with pytest.raises(
            ValueError,
            match='equivalence <= 0'
        ):
            experiments.TestMatrix(
                good_initial_pressure,
                good_initial_temperature,
                good_fuel,
                good_oxidizer,
                good_diluent,
                bad_equivalence,
                good_diluent_mole_fraction,
                good_num_replicates,
                good_tube_volume
            )

    # non-iterable numeric diluent_mole_fraction
    matrix = experiments.TestMatrix(
        good_initial_pressure,
        good_initial_temperature,
        good_fuel,
        good_oxidizer,
        good_diluent,
        good_equivalence,
        0.2,
        good_num_replicates,
        good_tube_volume
    )

    # check to make sure that matrix.diluent_mole_fraction is iterable
    assert len(matrix.diluent_mole_fraction) == 1

    # iterable diluent_mole_fraction with non-numeric values
    with pytest.raises(
        TypeError,
        match='diluent_mole_fraction has non-numeric items'
    ):
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_fuel,
            good_oxidizer,
            good_diluent,
            good_equivalence,
            'iterable and non numeric',
            good_num_replicates,
            good_tube_volume
        )

    # diluent_mole_fraction outside of [0, 1)
    for bad_diluent_mole_fraction in [-1., 7, 1]:
        with pytest.raises(
            ValueError,
            match='diluent mole fraction <0 or >= 1'
        ):
            experiments.TestMatrix(
                good_initial_pressure,
                good_initial_temperature,
                good_fuel,
                good_oxidizer,
                good_diluent,
                good_equivalence,
                bad_diluent_mole_fraction,
                good_num_replicates,
                good_tube_volume
            )

    # edge cases for diluent_mole_fraction
    for okay_diluent_mole_fraction in [0, 0.99]:
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_fuel,
            good_oxidizer,
            good_diluent,
            good_equivalence,
            okay_diluent_mole_fraction,
            good_num_replicates,
            good_tube_volume
        )

    # </editor-fold>

    # <editor-fold desc="_build_replicate()">
    ...
    # </editor-fold>

    # <editor-fold desc="generate_test_matrices()">
    ...
    # </editor-fold>

    # <editor-fold desc="save()">
    ...
    # </editor-fold>
