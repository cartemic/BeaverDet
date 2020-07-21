# -*- coding: utf-8 -*-

import os

import numpy as np
import pint
import pytest

from .. import thermochem

_U = pint.UnitRegistry()
_Q = _U.Quantity


class TestCalculateLaminarFlameSpeed:
    initial_temperature = _Q(300, "K")
    initial_pressure = _Q(1, "atm")

    def test_good_input(self):
        species = {
            "CH4": 0.095057034220532327,
            "O2": 0.19011406844106465,
            "N2": 0.71482889733840305
        }
        good_result = 0.39  # value approximated from Law fig. 7.7.7
        test_flame_speed = thermochem.calculate_laminar_flame_speed(
            self.initial_temperature,
            self.initial_pressure,
            species,
            "gri30.cti"
        )
        assert abs(test_flame_speed.magnitude - good_result) / \
            good_result < 0.05

    def test_no_species(self):
        species = {}
        with pytest.raises(
                ValueError,
                match="Empty species dictionary"
        ):
            thermochem.calculate_laminar_flame_speed(
                self.initial_temperature,
                self.initial_pressure,
                species,
                "gri30.cti"
            )

    def test_bad_species(self):
        species = {
            "Wayne": 3,
            "CH4": 7,
            "Garth": 5
        }
        with pytest.raises(
                ValueError,
                match="Species not in mechanism:\nWayne\nGarth\n"
        ):
            thermochem.calculate_laminar_flame_speed(
                self.initial_temperature,
                self.initial_pressure,
                species,
                "gri30.cti"
            )


class TestGetEqSoundSpeed:
    # check air at 1 atm and 20°C against ideal gas calculation
    gamma = 1.4
    rr = 8.31451
    tt = 293.15
    mm = 0.0289645
    c_ideal = np.sqrt(gamma * rr * tt / mm)

    temp = _Q(20, "degC")
    press = _Q(1, "atm")
    species = {"O2": 1, "N2": 3.76}
    mechanism = "gri30.cti"

    def test_no_unit_registry(self):
        c_test = thermochem.get_eq_sound_speed(
            self.temp,
            self.press,
            self.species,
            self.mechanism
        )

        assert abs(self.c_ideal - c_test.to("m/s").magnitude) / \
            self.c_ideal <= 0.005

    def test_unit_registry(self):
        c_test = thermochem.get_eq_sound_speed(
            self.temp,
            self.press,
            self.species,
            self.mechanism,
            unit_registry=_U
        )

        assert abs(self.c_ideal - c_test.to("m/s").magnitude) / \
            self.c_ideal <= 0.005


def test_calculate_reflected_shock_state():
    # this is just a handler for some sd2 functions, and this test is to ensure
    # that it doesn't throw any errors
    initial_temperature = _Q(80, "degF")
    initial_pressure = _Q(1, "atm")
    species_dict = {"H2": 1, "O2": 0.5}
    mechanism = "gri30.cti"
    thermochem.calculate_reflected_shock_state(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        _U
    )


class TestFindMechanisms:
    def test_mechs_only(self):
        assert "gri30.cti" in thermochem.find_mechanisms()

    def test_return_directory(self):
        checks = [False, False]
        mechs, path = thermochem.find_mechanisms(True)
        checks[0] = "gri30.cti" in mechs
        checks[1] = os.path.exists(path)
        assert all(checks)
