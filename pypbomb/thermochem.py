# -*- coding: utf-8 -*-
"""
This module contains functions for performing thermochemical calculations using
``cantera`` and ``pypbomb.sd``.
"""

import cantera as ct
import numpy as np
import pint

from . import tools, sd


_U = pint.UnitRegistry()


def calculate_laminar_flame_speed(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        phase_specification="",
        unit_registry=_U
):
    """
    This function uses cantera to calculate the laminar flame speed of a given
    gas mixture.

    Parameters
    ----------
    initial_temperature : pint.Quantity
        Initial temperature of gas mixture
    initial_pressure : pint.Quantity
        Initial pressure of gas mixture
    species_dict : dict
        Dictionary with species names (all caps) as keys and moles as values
    mechanism : str
        String of mechanism to use (e.g. "gri30.cti")
    phase_specification : str
        Phase specification for cantera solution
    unit_registry : pint.UnitRegistry
        Unit registry for managing units to prevent conflicts with parent
        unit registry

    Returns
    -------
    pint.Quantity
        Laminar flame speed in m/s as a pint quantity
    """
    gas = ct.Solution(mechanism, phase_specification)
    quant = unit_registry.Quantity

    tools.check_pint_quantity(
        initial_pressure,
        "pressure",
        ensure_positive=True
    )
    tools.check_pint_quantity(
        initial_temperature,
        "temperature",
        ensure_positive=True
    )

    # ensure species dict isn't empty
    if len(species_dict) == 0:
        raise ValueError("Empty species dictionary")

    # ensure all species are in the mechanism file
    bad_species = ""
    good_species = gas.species_names
    for species in species_dict:
        if species not in good_species:
            bad_species += species + "\n"
    if len(bad_species) > 0:
        raise ValueError("Species not in mechanism:\n" + bad_species)

    gas.TPX = (
        initial_temperature.to("K").magnitude,
        initial_pressure.to("Pa").magnitude,
        species_dict
    )

    # find laminar flame speed
    flame = ct.FreeFlame(gas)
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    flame.solve(loglevel=0)

    return quant(flame.u[0], "m/s")


# noinspection SpellCheckingInspection
def get_eq_sound_speed(
        temperature,
        pressure,
        species_dict,
        mechanism,
        phase_specification="",
        unit_registry=_U
):
    """
    Calculates the equilibrium speed of sound in a mixture

    Parameters
    ----------
    temperature : pint.Quantity
        Initial mixture temperature
    pressure : pint.Quantity
        Initial mixture pressure
    species_dict : dict
        Dictionary of mixture mole fractions
    mechanism : str
        Desired chemical mechanism
    phase_specification : str
        Phase specification for cantera solution
    unit_registry : pint.UnitRegistry
        Unit registry for managing units to prevent conflicts with parent
        unit registry

    Returns
    -------
    sound_speed : pint.Quantity
        local speed of sound in m/s
    """
    quant = unit_registry.Quantity

    tools.check_pint_quantity(
        pressure,
        "pressure",
        ensure_positive=True
    )

    tools.check_pint_quantity(
        temperature,
        "temperature",
        ensure_positive=True
    )

    working_gas = ct.Solution(mechanism, phase_specification)
    working_gas.TPX = [
        temperature.to("K").magnitude,
        pressure.to("Pa").magnitude,
        species_dict
        ]

    pressures = np.zeros(2)
    densities = np.zeros(2)

    # equilibrate gas at input conditions and collect pressure, density
    working_gas.equilibrate("TP")
    pressures[0] = working_gas.P
    densities[0] = working_gas.density

    # perturb pressure and equilibrate with constant P, s to get dp/drho|s
    pressures[1] = 1.0001 * pressures[0]
    working_gas.SP = working_gas.s, pressures[1]
    working_gas.equilibrate("SP")
    densities[1] = working_gas.density

    # calculate sound speed
    sound_speed = np.sqrt(np.diff(pressures)/np.diff(densities))[0]

    return quant(sound_speed, "m/s")


def calculate_reflected_shock_state(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        unit_registry=_U,
        use_multiprocessing=False
):
    """
    Calculates the thermodynamic and chemical state of a reflected shock
    using customized sdtoolbox functions.

    Parameters
    ----------
    initial_temperature : pint.Quantity
        Pint quantity of mixture initial temperature
    initial_pressure : pint.Quantity
        Pint quantity of mixture initial pressure
    species_dict : dict
        Dictionary of initial reactant mixture
    mechanism : str
        Mechanism to use for chemical calculations, e.g. "gri30.cti"
    unit_registry : pint.UnitRegistry
        Pint unit registry
    use_multiprocessing : bool
        True to use multiprocessing for CJ state calculation, which is faster
        but requires the function to be run from __main__

    Returns
    -------
    dict
        Dictionary containing keys "reflected" and "cj". Each of these
        contains "speed", indicating the related wave speed, and "state",
        which is a Cantera gas object at the specified state.
    """
    quant = unit_registry.Quantity

    # define gas objects
    initial_gas = ct.Solution(mechanism)
    reflected_gas = ct.Solution(mechanism)

    # define gas states
    initial_temperature = initial_temperature.to("K").magnitude
    initial_pressure = initial_pressure.to("Pa").magnitude

    initial_gas.TPX = [
        initial_temperature,
        initial_pressure,
        species_dict
    ]
    reflected_gas.TPX = [
        initial_temperature,
        initial_pressure,
        species_dict
    ]

    # get CJ state
    cj_calcs = sd.Detonation.cj_speed(
        initial_pressure,
        initial_temperature,
        species_dict,
        mechanism,
        return_state=True,
        use_multiprocessing=use_multiprocessing
    )

    # get reflected state
    [_,
     reflected_speed,
     reflected_gas] = sd.Reflection.reflect(
        initial_gas,
        cj_calcs["cj state"],
        reflected_gas,
        cj_calcs["cj speed"]
    )

    return {
        "reflected": {
            "speed": quant(
                reflected_speed,
                "m/s"
            ),
            "state": reflected_gas
        },
        "cj": {
            "speed": quant(
                cj_calcs["cj speed"],
                "m/s"),
            "state": cj_calcs["cj state"]
        }
    }
