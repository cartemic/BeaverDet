# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for thermochem.py

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


class TestCalculateLaminarFlamespeed:
    initial_temperature = quant(300, 'K')
    initial_pressure = quant(1, 'atm')

    def test_good_input(self):
        species = {
            'CH4': 0.095057034220532327,
            'O2': 0.19011406844106465,
            'N2': 0.71482889733840305
        }
        good_result = 0.39  # value approximated from Law fig. 7.7.7
        test_flamespeed = thermochem.calculate_laminar_flamespeed(
            self.initial_temperature,
            self.initial_pressure,
            species,
            'gri30.cti'
        )
        assert abs(test_flamespeed.magnitude - good_result) / good_result < 0.05

    def test_no_species(self):
        species = {}
        with pytest.raises(
                ValueError,
                match='Empty species dictionary'
        ):
            thermochem.calculate_laminar_flamespeed(
                self.initial_temperature,
                self.initial_pressure,
                species,
                'gri30.cti'
            )

    def test_bad_species(self):
        species = {
            'Wayne': 3,
            'CH4': 7,
            'Garth': 5
        }
        with pytest.raises(
                ValueError,
                match='Species not in mechanism:\nWayne\nGarth\n'
        ):
            thermochem.calculate_laminar_flamespeed(
                self.initial_temperature,
                self.initial_pressure,
                species,
                'gri30.cti'
            )


class TestGetEqSoundSpeed:
    # check air at 1 atm and 20°C against ideal gas calculation
    gamma = 1.4
    rr = 8.31451
    tt = 293.15
    mm = 0.0289645
    c_ideal = np.sqrt(gamma * rr * tt / mm)

    temp = quant(20, 'degC')
    press = quant(1, 'atm')
    species = {'O2': 1, 'N2': 3.76}
    mechanism = 'gri30.cti'

    def test_no_unit_registry(self):
        c_test = thermochem.get_eq_sound_speed(
            self.temp,
            self.press,
            self.species,
            self.mechanism
        )

        assert abs(self.c_ideal - c_test.to('m/s').magnitude) / \
            self.c_ideal <= 0.005

    def test_unit_registry(self):
        c_test = thermochem.get_eq_sound_speed(
            self.temp,
            self.press,
            self.species,
            self.mechanism,
            unit_registry=ureg
        )

        assert abs(self.c_ideal - c_test.to('m/s').magnitude) / \
            self.c_ideal <= 0.005


def test_calculate_reflected_shock_state():
    # this is just a handler for some sdtoolbox functions, and this test is to
    # ensure that it doesn't throw any errors
    initial_temperature = quant(80, 'degF')
    initial_pressure = quant(1, 'atm')
    species_dict = {'H2': 1, 'O2': 0.5}
    mechanism = 'gri30.cti'
    thermochem.calculate_reflected_shock_state(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        ureg
    )


# noinspection PyProtectedMember
class TestMixture:
    initial_pressure = quant(1, 'atm')
    initial_temperature = quant(20, 'degC')
    good_fuel = 'H2'
    good_oxidizer = 'O2'
    good_diluent = 'AR'
    good_volume = quant(0.1028, "m^3")

    def test_init_bad_fuel(self):
        with pytest.raises(
            ValueError,
            match='Bad fuel'
        ):
            thermochem.Mixture(
                self.initial_pressure,
                self.initial_temperature,
                'a banana',
                self.good_oxidizer,
                self.good_diluent
            )

    def test_init_bad_oxidizer(self):
        with pytest.raises(
            ValueError,
            match='Bad oxidizer'
        ):
            thermochem.Mixture(
                self.initial_pressure,
                self.initial_temperature,
                self.good_fuel,
                'seven spatulas',
                self.good_diluent
            )

    def test_init_bad_diluent(self):
        with pytest.raises(
            ValueError,
            match='Bad diluent'
        ):
            thermochem.Mixture(
                self.initial_pressure,
                self.initial_temperature,
                self.good_fuel,
                self.good_oxidizer,
                'dog poop'
            )

    def test_init_with_dilution(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer,
            self.good_diluent,
            diluent_mole_fraction=0.1
        )
        assert test_mixture.diluted is not None

    def test_add_diluent_bad_diluent(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )

        bad_diluent = 'pinot noir'

        with pytest.raises(
            ValueError,
            match='Bad diluent: ' + bad_diluent
        ):
            test_mixture.add_diluent(
                diluent=bad_diluent,
                mole_fraction=0.1
            )

    def test_add_diluent_fuel_oxidizer_diluent(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )

        bad_diluents = [self.good_fuel, self.good_oxidizer]

        for bad_diluent in bad_diluents:
            with pytest.raises(
                ValueError,
                match='You can\'t dilute with fuel or oxidizer!'
            ):
                test_mixture.add_diluent(
                    diluent=bad_diluent,
                    mole_fraction=0.1
                )

    def test_add_diluent_bad_mole_fraction(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )

        bad_mole_fractions = [-2, 1.7]

        for bad_mole_fraction in bad_mole_fractions:
            with pytest.raises(
                ValueError,
                match='Bro, do you even mole fraction?'
            ):
                test_mixture.add_diluent(
                    diluent=self.good_diluent,
                    mole_fraction=bad_mole_fraction
                )

    def test_get_masses_diluted(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        test_mixture.add_diluent(
            self.good_diluent,
            0.2
        )
        good_masses = {
            self.good_fuel: 0.004578,
            self.good_oxidizer: 0.03633,
            self.good_diluent: 0.34017
        }

        test_masses = {
            key: value.to('kg').magnitude
            for key, value in
            test_mixture.get_masses(
                self.good_volume,
                diluted=True
            ).items()
        }

        key_check = list(test_masses.keys()) == list(good_masses.keys())
        value_check = [
            np.allclose(test_value, good_value)
            for test_value, good_value in
            zip(
                list(test_masses.values()),
                list(good_masses.values())
            )
        ]

        assert all([key_check, value_check])

    def test_get_masses_diluted_without_dilution(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        with pytest.raises(
            ValueError,
            match='Mixture has not been diluted'
        ):
            test_mixture.get_masses(
                self.good_volume,
                diluted=True
            )

    def test_get_masses_undiluted(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        good_masses = {
            self.good_fuel: 0.00572,
            self.good_oxidizer: 0.04541
        }

        test_masses = {
            key: value.to('kg').magnitude
            for key, value in
            test_mixture.get_masses(
                self.good_volume,
                diluted=False
            ).items()
        }

        key_check = list(test_masses.keys()) == list(good_masses.keys())
        value_check = [
            np.allclose(test_value, good_value)
            for test_value, good_value in
            zip(
                list(test_masses.values()),
                list(good_masses.values())
            )
        ]

        assert all([key_check, value_check])

    def test_get_pressures_diluted(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        test_mixture.add_diluent(
            self.good_diluent,
            0.2
        )
        good_pressures = {
            self.good_fuel: 54040,
            self.good_oxidizer: 27020,
            self.good_diluent: 20265
        }

        test_pressures = {
            key: value.to('Pa').magnitude
            for key, value in
            test_mixture.get_pressures(diluted=True).items()
        }

        key_check = list(test_pressures.keys()) == list(good_pressures.keys())
        value_check = [
            np.allclose(test_value, good_value)
            for test_value, good_value in
            zip(
                list(test_pressures.values()),
                list(good_pressures.values())
            )
        ]

        assert all([key_check, value_check])

    def test_get_pressures_diluted_without_dilution(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        with pytest.raises(
            ValueError,
            match='Mixture has not been diluted'
        ):
            test_mixture.get_pressures(diluted=True)

    def test_get_pressures_undiluted(self):
        test_mixture = thermochem.Mixture(
            self.initial_pressure,
            self.initial_temperature,
            self.good_fuel,
            self.good_oxidizer
        )
        good_pressures = {
            self.good_fuel: 67550,
            self.good_oxidizer: 33775
        }

        test_pressures = {
            key: value.to('Pa').magnitude
            for key, value in
            test_mixture.get_pressures(diluted=False).items()
        }

        key_check = list(test_pressures.keys()) == list(good_pressures.keys())
        value_check = [
            np.allclose(test_value, good_value)
            for test_value, good_value in
            zip(
                list(test_pressures.values()),
                list(good_pressures.values())
            )
        ]

        assert all([key_check, value_check])

    # def test_mixed_oxidizer(self):
    #     test_mixture = thermochem.Mixture(
    #         self.initial_pressure,
    #         self.initial_temperature,
    #         self.good_fuel,
    #         'O2:1 N2:3.76'
    #     )
    #     test_mixture.get_masses(
    #             self.good_volume,
    #             diluted=False
    #         )
    #     # TODO: write tests for mixed fuel, oxidizer, diluent
    #     assert False
