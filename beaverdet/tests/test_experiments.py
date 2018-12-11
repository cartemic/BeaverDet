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
current_dir = os.path.join(
    *os.path.split(
        os.path.dirname(__file__)
    )[:-1]
)
test_data_dir = os.path.join(
    current_dir,
    'tests',
    'test_data'
)


class TestTestMatrix:
    good_initial_pressure = quant(1, 'atm')
    good_initial_temperature = quant(70, 'degF')
    good_equivalence = [1]
    good_diluent_mole_fraction = [0]
    good_num_replicates = 3
    good_tube_volume = quant(0.1028, 'm^3')
    good_fuel = 'H2'
    good_oxidizer = 'O2'
    good_diluent = 'AR'

    def test_initialize_negative_replicates(self):
        # num_replicates <= 0
        with pytest.raises(
            ValueError,
            match='bad number of replicates'
        ):
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                self.good_equivalence,
                self.good_diluent_mole_fraction,
                0,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                self.good_diluent
            )

    def test_initialize_non_iterable_equivalence(self):
        # non-iterable numeric equivalence
        matrix = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            1,
            self.good_diluent_mole_fraction,
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )

        # check to make sure that matrix.equivalence is iterable
        assert len(matrix.equivalence) == 1

    def test_initialize_non_numeric_equivalence(self):
        # iterable equivalence with non-numeric values
        with pytest.raises(
            TypeError,
            match='equivalence has non-numeric items'
        ):
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                'iterable and non numeric',
                self.good_diluent_mole_fraction,
                self.good_num_replicates,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                self.good_diluent
            )

    def test_initialize_negative_equivalence(self):
        # equivalence <= 0
        for bad_equivalence in [[0], [-1]]:
            with pytest.raises(
                ValueError,
                match='equivalence <= 0'
            ):
                experiments.TestMatrix(
                    self.good_initial_pressure,
                    self.good_initial_temperature,
                    bad_equivalence,
                    self.good_diluent_mole_fraction,
                    self.good_num_replicates,
                    self.good_tube_volume,
                    self.good_fuel,
                    self.good_oxidizer,
                    self.good_diluent
                )

    def test_initialize_non_iterable_diluent_mf(self):
        # non-iterable numeric diluent_mole_fraction
        matrix = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            self.good_equivalence,
            0.2,
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )

        # check to make sure that matrix.diluent_mole_fraction is iterable
        assert len(matrix.diluent_mole_fraction) == 1

    def test_initialize_non_numeric_diluent_mf(self):
        # iterable diluent_mole_fraction with non-numeric values
        with pytest.raises(
            TypeError,
            match='diluent_mole_fraction has non-numeric items'
        ):
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                self.good_equivalence,
                'iterable and non numeric',
                self.good_num_replicates,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                self.good_diluent
            )

    def test_initialize_diluent_mf_out_of_range(self):
        # diluent_mole_fraction outside of [0, 1)
        for bad_diluent_mole_fraction in [-1., 7, 1]:
            with pytest.raises(
                ValueError,
                match='diluent mole fraction <0 or >= 1'
            ):
                experiments.TestMatrix(
                    self.good_initial_pressure,
                    self.good_initial_temperature,
                    self.good_equivalence,
                    bad_diluent_mole_fraction,
                    self.good_num_replicates,
                    self.good_tube_volume,
                    self.good_fuel,
                    self.good_oxidizer,
                    self.good_diluent
                )

    def test_initialize_diluent_mf_edge_cases(self):
        # edge cases for diluent_mole_fraction
        for okay_diluent_mole_fraction in [0, 0.99]:
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                self.good_equivalence,
                okay_diluent_mole_fraction,
                self.good_num_replicates,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                self.good_diluent
            )

    def test_undiluted(self):
        undiluted = [
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                [0.75, 1],
                [0.2],
                self.good_num_replicates,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                diluent=None
            ),
            experiments.TestMatrix(
                self.good_initial_pressure,
                self.good_initial_temperature,
                [0.75, 1],
                [0],
                self.good_num_replicates,
                self.good_tube_volume,
                self.good_fuel,
                self.good_oxidizer,
                self.good_diluent
            )
            ]
        undiluted_correct_path = os.path.join(
            test_data_dir,
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

    def test_diluted(self):
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0.1, 0.2],
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )
        diluted_correct_path = os.path.join(
            test_data_dir,
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

    def test_replicates_not_none(self):
        # make sure mixture.replicates contains no None
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0.1, 0.2],
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )
        diluted.generate_test_matrices()
        assert all(item is not None for item in diluted.replicates)

    def test_replicate_randomization(self):
        # make sure replicates are randomized
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0.1, 0.2],
            10,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )
        diluted.generate_test_matrices()

        replicates_different = []
        for replicate in diluted.replicates[1:]:
            replicates_different.append(
                not diluted.replicates[0].equals(replicate)
            )
        assert any(replicates_different)

    def test_save_with_dilution(self):
        # make sure test matrix save works as planned
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0.1, 0.2],
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )
        diluted.generate_test_matrices()
        # collect list of files in directory
        start_files = os.listdir(test_data_dir)

        diluted.save(test_data_dir)
        end_files = os.listdir(test_data_dir)
        delete_files = [file for file in end_files if file not in start_files]
        print(start_files)
        print(end_files)

        assert len(delete_files) == self.good_num_replicates

        [os.remove(os.path.join(test_data_dir, file)) for file in
         delete_files]

    def test_save_without_dilution(self):
        # make sure test matrix save works as planned
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0],
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            None
        )
        diluted.generate_test_matrices()

        # collect list of files in directory
        start_files = os.listdir(test_data_dir)

        diluted.save(test_data_dir)
        end_files = os.listdir(test_data_dir)
        delete_files = [file for file in end_files if file not in start_files]
        print(start_files)
        print(end_files)

        assert len(delete_files) == self.good_num_replicates

        [os.remove(os.path.join(test_data_dir, file)) for file in
         delete_files]

    def test_save_without_generation(self):
        # make sure test matrix save works as planned
        diluted = experiments.TestMatrix(
            self.good_initial_pressure,
            self.good_initial_temperature,
            [0.75, 1],
            [0.1, 0.2],
            self.good_num_replicates,
            self.good_tube_volume,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent
        )

        # collect list of files in directory
        start_files = os.listdir(test_data_dir)

        diluted.save(test_data_dir)
        end_files = os.listdir(test_data_dir)
        delete_files = [file for file in end_files if file not in start_files]
        print(start_files)
        print(end_files)

        assert len(delete_files) == self.good_num_replicates

        [os.remove(os.path.join(test_data_dir, file)) for file in
         delete_files]
