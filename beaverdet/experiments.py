# -*- coding: utf-8 -*-
"""
PURPOSE:
    Tools for conducting detonation experiments and analyzing the data

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import pandas as pd
import datetime
import os
from . import tools, thermochem


class TestMatrix:
    def __init__(
            self,
            initial_pressure,
            initial_temperature,
            equivalence,
            diluent_mole_fraction,
            num_replicates,
            tube_volume,
            fuel,
            oxidizer,
            diluent=None
    ):
        tools.check_pint_quantity(
            initial_pressure,
            'pressure'
        )
        tools.check_pint_quantity(
            initial_temperature,
            'temperature'
        )
        tools.check_pint_quantity(
            tube_volume,
            'volume'
        )

        # enforce num_replicates as int, and ensure > 0
        num_replicates = int(num_replicates)
        if num_replicates <= 0:
            raise ValueError('bad number of replicates')

        # ensure that equivalence is iterable
        try:
            len(equivalence)
        except TypeError:
            # make equivalence iterable
            equivalence = [equivalence]
        try:
            # check that all items are numeric
            [item / 7 for item in equivalence]
        except TypeError:
            raise TypeError('equivalence has non-numeric items')

        # ensure equivalence >= 0
        for test_equivalence in equivalence:
            if test_equivalence <= 0:
                raise ValueError('equivalence <= 0')

        self.equivalence = equivalence

        # repeat for diluent mole fraction
        try:
            len(diluent_mole_fraction)
        except TypeError:
            # make diluent_mole_fraction iterable
            diluent_mole_fraction = [diluent_mole_fraction]
        try:
            # check that all items are numeric
            [item / 7 for item in diluent_mole_fraction]
        except TypeError:
            raise TypeError('diluent_mole_fraction has non-numeric items')

        # make sure mixture is undiluted if all mole fractions are zero
        if all(mole_fraction == 0 for mole_fraction in diluent_mole_fraction):
            diluent = None
        self.diluent = diluent

        self.mixture = thermochem.Mixture(
            initial_pressure,
            initial_temperature,
            fuel,
            oxidizer,
            diluent=diluent
        )
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.tube_volume = tube_volume
        self.base_replicate = None
        self.replicates = [
            None for _ in range(num_replicates)
        ]

        # ensure diluent mole fraction is on [0, 1)
        for test_mole_fraction in diluent_mole_fraction:
            if (test_mole_fraction < 0) or (test_mole_fraction >= 1):
                raise ValueError('diluent mole fraction <0 or >= 1')

        self.diluent_mole_fraction = diluent_mole_fraction

    def _build_replicate(
            self
    ):
        # initialize dataframe
        current_replicate = pd.DataFrame(
            columns=[
                'Equivalence',
                'Diluent Mol Fraction',
                '{} (Pa)'.format(self.diluent),
                self.oxidizer + ' (Pa)',
                self.fuel + ' (Pa)',
                '{} (kg)'.format(self.diluent),
                self.oxidizer + ' (kg)',
                self.fuel + ' (kg)'
            ]
        )

        for phi in self.equivalence:
            # set mixture undiluted equivalence ratio
            self.mixture.set_equivalence(phi)

            for dil_mf in self.diluent_mole_fraction:
                if self.diluent and dil_mf:
                    # dilute mixture
                    self.mixture.add_diluent(
                        self.diluent,
                        dil_mf
                    )

                    # calculate partial pressures and total masses of used gas
                    fill_pressure = self.mixture.diluted.P  # Pascals
                    partials = {
                        gas: mol_fraction * fill_pressure for gas, mol_fraction
                        in self.mixture.diluted.mole_fraction_dict().items()
                    }
                    masses = self.mixture.get_mass(
                        self.tube_volume,
                        diluted=True
                    )

                    # build test matrix row
                    current_row = [
                        phi,
                        dil_mf,
                        partials[self.diluent],
                        (
                            partials[self.diluent] +
                            partials[self.oxidizer]
                        ),
                        (
                            partials[self.diluent] +
                            partials[self.oxidizer] +
                            partials[self.fuel]
                        ),
                        masses[self.diluent].to('kg').magnitude,
                        masses[self.oxidizer].to('kg').magnitude,
                        masses[self.fuel].to('kg').magnitude
                    ]

                    # add row to dataframe
                    tools.add_dataframe_row(
                        current_replicate,
                        current_row
                    )
                else:
                    # calculate partial pressures and total masses of used gas
                    fill_pressure = self.mixture.undiluted.P  # Pascals
                    partials = {
                        gas: mol_fraction * fill_pressure for gas, mol_fraction
                        in self.mixture.undiluted.mole_fraction_dict().items()
                    }
                    masses = self.mixture.get_mass(
                        self.tube_volume,
                        diluted=False
                    )

                    # build test matrix row
                    current_row = [
                        phi,
                        0,  # diluent mole fraction
                        0,  # diluent fill pressure
                        partials[self.oxidizer],
                        (
                                partials[self.oxidizer] +
                                partials[self.fuel]
                        ),
                        0,  # diluent mass (kg)
                        masses[self.oxidizer].to('kg').magnitude,
                        masses[self.fuel].to('kg').magnitude
                    ]

                    # add row to dataframe
                    tools.add_dataframe_row(
                        current_replicate,
                        current_row
                    )

        return current_replicate

    def generate_test_matrices(
            self
    ):
        self.base_replicate = self._build_replicate()
        self.replicates = [
            self.base_replicate.copy().sample(frac=1).reset_index(drop=True) for
            _ in self.replicates
        ]

    def save(
            self,
            directory
    ):
        # make sure that replicates have been generated
        if any([item is None for item in self.replicates]):
            self.generate_test_matrices()

        # avoid weirdness when self.diluent=None
        if not self.diluent:
            diluent = ''
        else:
            diluent = self.diluent + '_'

        for num, replicate in enumerate(self.replicates):
            # build file save path and export to .csv
            now = datetime.datetime.now()
            file_name = (
                    '{0}{1:02}{2}_{3:02}{4:02}{5:02}_{6}_{7}_{8}replicate_' +
                    '{9:02}.csv'
            ).format(
                now.year,
                now.month,
                now.day,
                now.hour,
                now.minute,
                now.second,
                self.fuel,
                self.oxidizer,
                diluent,
                num
            )
            save_path = os.path.join(
                directory,
                file_name
            )
            self.replicates[num].to_csv(save_path, index=False)
