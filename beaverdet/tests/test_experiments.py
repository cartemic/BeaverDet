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
import os
import pandas as pd
import numpy as np
from .. import experiments

ureg = pint.UnitRegistry()
quant = ureg.Quantity


def compare_dataframe():
    pass


def test_test_matrix():
    good_initial_pressure = quant(1, 'atm')
    good_initial_temperature = quant(70, 'degF')
    good_equivalence = [1]
    good_diluent_mole_fraction = [0]
    good_num_replicates = 3
    good_tube_volume = quant(0.1028, 'm^3')
    good_fuel = 'H2'
    good_oxidizer = 'O2'
    good_diluent = 'AR'

    # <editor-fold desc="__init__()">
    # num_replicates <= 0
    with pytest.raises(
        ValueError,
        match='bad number of replicates'
    ):
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_equivalence,
            good_diluent_mole_fraction,
            0,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            good_diluent
        )

    # non-iterable numeric equivalence
    matrix = experiments.TestMatrix(
        good_initial_pressure,
        good_initial_temperature,
        1,
        good_diluent_mole_fraction,
        good_num_replicates,
        good_tube_volume,
        good_fuel,
        good_oxidizer,
        good_diluent
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
            'iterable and non numeric',
            good_diluent_mole_fraction,
            good_num_replicates,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            good_diluent
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
                bad_equivalence,
                good_diluent_mole_fraction,
                good_num_replicates,
                good_tube_volume,
                good_fuel,
                good_oxidizer,
                good_diluent
            )

    # non-iterable numeric diluent_mole_fraction
    matrix = experiments.TestMatrix(
        good_initial_pressure,
        good_initial_temperature,
        good_equivalence,
        0.2,
        good_num_replicates,
        good_tube_volume,
        good_fuel,
        good_oxidizer,
        good_diluent
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
            good_equivalence,
            'iterable and non numeric',
            good_num_replicates,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            good_diluent
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
                good_equivalence,
                bad_diluent_mole_fraction,
                good_num_replicates,
                good_tube_volume,
                good_fuel,
                good_oxidizer,
                good_diluent
            )

    # edge cases for diluent_mole_fraction
    for okay_diluent_mole_fraction in [0, 0.99]:
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            good_equivalence,
            okay_diluent_mole_fraction,
            good_num_replicates,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            good_diluent
        )

    # </editor-fold>

    # <editor-fold desc="_build_replicate()/generate_test_matrices()">
    undiluted = [
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            [0.75, 1],
            [0.2],
            good_num_replicates,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            diluent=None
        ),
        experiments.TestMatrix(
            good_initial_pressure,
            good_initial_temperature,
            [0.75, 1],
            [0],
            good_num_replicates,
            good_tube_volume,
            good_fuel,
            good_oxidizer,
            good_diluent
        )
        ]
    undiluted_correct_path = os.path.join(
        os.path.abspath(os.path.curdir),
        'beaverdet',
        'tests',
        'test_data',
        'undiluted_test.csv'
    )
    undiluted_correct = pd.read_csv(
        undiluted_correct_path,
        dtype=np.float64
    )

    for mixture in undiluted:
        mixture.generate_test_matrices()
        pd.testing.assert_frame_equal(
            mixture.base_replicate,
            undiluted_correct
        )

    diluted = experiments.TestMatrix(
        good_initial_pressure,
        good_initial_temperature,
        [0.75, 1],
        [0.1, 0.2],
        good_num_replicates,
        good_tube_volume,
        good_fuel,
        good_oxidizer,
        good_diluent
    )
    diluted_correct_path = os.path.join(
        os.path.abspath(os.path.curdir),
        'beaverdet',
        'tests',
        'test_data',
        'diluted_test.csv'
    )
    diluted_correct = pd.read_csv(
        diluted_correct_path,
        dtype=np.float64
    )
    diluted.generate_test_matrices()
    pd.testing.assert_frame_equal(
        diluted.base_replicate,
        diluted_correct
    )

    # make sure mixture.replicates contains no None
    assert all(item is not None for item in diluted.replicates)

    # make sure replicates are randomized
    assert not diluted.replicates[0].equals(diluted.replicates[1])

    # </editor-fold>/gen

    # <editor-fold desc="save()">
    test_save_directory = os.path.join(
        os.path.abspath(os.path.curdir),
        'beaverdet',
        'tests',
        'test_data'
    )
    # collect list of files in directory
    start_files = os.listdir(test_save_directory)

    diluted.save(test_save_directory)
    end_files = os.listdir(test_save_directory)
    delete_files = [file for file in end_files if file not in start_files]
    print(start_files)
    print(end_files)

    assert len(delete_files) == good_num_replicates

    [os.remove(os.path.join(test_save_directory, file)) for file in
     delete_files]
    # </editor-fold>
