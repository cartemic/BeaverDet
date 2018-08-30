# -*- coding: utf-8 -*-
"""
PURPOSE:
    Thermochemical calculations

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import pint
import numpy as np
import cantera as ct
from . import tools


def calculate_laminar_flamespeed(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        phase_specification='',
        unit_registry=None
):
    """
    This function uses cantera to calculate the laminar flame speed of a given
    gas mixture.

    Parameters
    ----------
    initial_temperature : pint quantity
        Initial temperature of gas mixture
    initial_pressure : pint quantity
        Initial pressure of gas mixture
    species_dict : dict
        Dictionary with species names (all caps) as keys and moles as values
    mechanism : str
        String of mechanism to use (e.g. 'gri30.cti')
    phase_specification : str
        Phase specification for cantera solution
    unit_registry : pint unit registry
        Unit registry for managing units to prevent conflicts with parent
        unit registry

    Returns
    -------
    Laminar flame speed in m/s as a pint quantity
    """
    gas = ct.Solution(mechanism, phase_specification)

    if not unit_registry:
        unit_registry = pint.UnitRegistry()
    quant = unit_registry.Quantity

    tools.check_pint_quantity(
        initial_pressure,
        'pressure',
        ensure_positive=True
    )
    tools.check_pint_quantity(
        initial_temperature,
        'temperature',
        ensure_positive=True
    )

    # ensure species dict isn't empty
    if len(species_dict) == 0:
        raise ValueError('Empty species dictionary')

    # ensure all species are in the mechanism file
    bad_species = ''
    good_species = gas.species_names
    for species in species_dict:
        if species not in good_species:
            bad_species += species + '\n'
    if len(bad_species) > 0:
        raise ValueError('Species not in mechanism:\n' + bad_species)

    gas.TPX = (
        initial_temperature.to('K').magnitude,
        initial_pressure.to('Pa').magnitude,
        species_dict
    )

    # find laminar flame speed
    flame = ct.FreeFlame(gas)
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    flame.solve(loglevel=0)

    return quant(flame.u[0], 'm/s')


def get_equil_sound_speed(
        temperature,
        pressure,
        species_dict,
        mechanism,
        phase_specification='',
        unit_registry=None
):
    """
    Calculates the equilibrium speed of sound in a mixture

    Parameters
    ----------
    temperature : pint quantity
        Initial mixture temperature
    pressure : pint quantity
        Initial mixture pressure
    species_dict : dict
        Dictionary of mixture mole fractions
    mechanism : str
        Desired chemical mechanism
    phase_specification : str
        Phase specification for cantera solution
    unit_registry : pint unit registry
        Unit registry for managing units to prevent conflicts with parent
        unit registry

    Returns
    -------
    sound_speed : pint quantity
        local speed of sound in m/s
    """
    if not unit_registry:
        unit_registry = pint.UnitRegistry()

    quant = unit_registry.Quantity

    tools.check_pint_quantity(
        pressure,
        'pressure',
        ensure_positive=True
    )

    tools.check_pint_quantity(
        temperature,
        'temperature',
        ensure_positive=True
    )

    working_gas = ct.Solution(mechanism, phase_specification)
    working_gas.TPX = [
        temperature.to('K').magnitude,
        pressure.to('Pa').magnitude,
        species_dict
        ]

    pressures = np.zeros(2)
    densities = np.zeros(2)

    # equilibrate gas at input conditions and collect pressure, density
    working_gas.equilibrate('TP')
    pressures[0] = working_gas.P
    densities[0] = working_gas.density

    # perturb pressure and equilibrate with constant P, s to get dp/drho|s
    pressures[1] = 1.0001 * pressures[0]
    working_gas.SP = working_gas.s, pressures[1]
    working_gas.equilibrate('SP')
    densities[1] = working_gas.density

    # calculate sound speed
    sound_speed = np.sqrt(np.diff(pressures)/np.diff(densities))[0]

    return quant(sound_speed, 'm/s')


# TODO: IN WORK
class Detonation:
    def __init__(
            self,
            initial_pressure,
            initial_temperature,
            fuel,
            oxidizer,
            diluent=None,
            equivalence=1,
            diluent_mole_fraction=0,
            mechanism='gri30.cti',
            unit_registry=None
    ):
        if not unit_registry:
            unit_registry = pint.UnitRegistry()
        self._quant = unit_registry.Quantity

        tools.check_pint_quantity(
            initial_pressure,
            'pressure',
            ensure_positive=True
        )

        tools.check_pint_quantity(
            initial_temperature,
            'temperature',
            ensure_positive=True
        )

        # make sure the user input species that are in the mechanism file
        good_species = self.undiluted.species_names
        if fuel in good_species:
            self.fuel = fuel
        else:
            raise ValueError('Bad fuel')
        if oxidizer in good_species:
            self.oxidizer = oxidizer
        else:
            raise ValueError('Bad oxidizer')
        if (diluent and diluent in good_species) or not diluent:
            self.diluent = diluent
        else:
            raise ValueError('Bad diluent')

        # define givens
        self.mechanism = mechanism
        self.initial_pressure = initial_pressure
        self.initial_temperature = initial_temperature
        self.diluent_mol_fraction = diluent_mole_fraction

        # initialize diluted and undiluted gas solution in Cantera
        self.undiluted = ct.Solution(mechanism)
        self.undiluted.TP = (
            self.initial_temperature.to('degK').magnitude,
            self.initial_pressure.to('Pa').magnitude
        )

        # set equivalence ratio
        self.equivalence = None
        self.set_equivalence(equivalence)

        # initialize diluted gas solution if diluent and mass fraction are
        # defined
        if diluent and diluent_mole_fraction:
            self.diluted = ct.Solution(mechanism)
            self.diluted.TP = (
                self.initial_temperature.to('degK').magnitude,
                self.initial_pressure.to('Pa').magnitude
            )
        else:
            self.diluted = None

    def set_equivalence(
            self,
            equivalence_ratio
    ):
        """
        Sets the equivalence ratio of the undiluted mixture using Cantera
        """
        equivalence_ratio = float(equivalence_ratio)

        # set the equivalence ratio
        self.undiluted.set_equivalence_ratio(equivalence_ratio,
                                             self.fuel,
                                             self.oxidizer)
        try:
            self.add_diluent(self.diluent, self.diluent_mol_fraction)
        except AttributeError:
            pass

        # ensure good inputs were given and record new equivalence ratio
        if sum([self.undiluted.X > 0][0]) < 2:
            raise ValueError('You can\'t detonate that, ya dingus')
        self.equivalence = equivalence_ratio

    def get_mixture_string(
            self,
            diluted=False
    ):
        """
        Gets a mixture string from either the diluted or undiluted Cantera
        solution object, which is then used to calculate CJ velocity using
        SDToolbox
        """
        if diluted:
            cantera_solution = self.diluted
        else:
            cantera_solution = self.undiluted
        mixture_list = []
        for i, species in enumerate(cantera_solution.species_names):
            if cantera_solution.X[i] > 0:
                mixture_list.append(species + ':' +
                                    str(cantera_solution.X[i]))
        return ' '.join(mixture_list)

    def add_diluent(self, diluent, mole_fraction):
        """
        Adds a diluent to an undiluted mixture, keeping the same equivalence
        ratio.
        """
        # make sure diluent is available in mechanism and isn't the fuel or ox
        if diluent not in self.undiluted.species_names:
            raise ValueError('Bad diluent:', diluent)
        elif diluent in [self.fuel, self.oxidizer]:
            raise ValueError('You can\'t dilute with fuel or oxidizer!')
        elif mole_fraction > 1.:
            raise ValueError('Bro, do you even mole fraction?')

        self.diluent = diluent
        self.diluent_mol_fraction = mole_fraction

        # collect undiluted mole fractions
        mole_fractions = self.undiluted.mole_fraction_dict()

        # add diluent and adjust mole fractions so they sum to 1
        new_fuel = (1 - mole_fraction) * mole_fractions[self.fuel]
        new_ox = (1 - mole_fraction) * mole_fractions[self.oxidizer]
        species = '{0}: {1} {2}: {3} {4}: {5}'.format(
            diluent,
            mole_fraction,
            self.fuel,
            new_fuel,
            self.oxidizer,
            new_ox
        )
        try:
            # add to diluted cantera solution
            self.diluted.X = species
        except AttributeError:
            # create cantera solution if one doesn't exist
            self.diluted = ct.Solution(self.mechanism)
            self.diluted.TPX = (
                self.initial_temperature.to('degK').magnitude,
                self.initial_pressure.to('Pa').magnitude,
                species
            )

    def get_mass(
            self,
            tube_volume,
            diluted=False
    ):
        """
        The cantera concentration function is used to collect
        species concentrations, in kmol/m^3, which are then multiplied by
        the molecular weights in kg/kmol to get the density in kg/m^3. This
        is then multiplied by the tube volume to get the total mass of each
        component.
        """
        tools.check_pint_quantity(
            tube_volume,
            'volume',
            ensure_positive=True
        )

        if diluted:
            cantera_solution = self.diluted
        else:
            cantera_solution = self.undiluted
        mixture_list = []
        for i, species in enumerate(cantera_solution.species_names):
            if cantera_solution.X[i] > 0:
                r_specific = self._quant(
                    8314 / cantera_solution.molecular_weights[i],
                    'J/(kg*K)'
                )
                pressure = self.initial_pressure * cantera_solution.X[i]
                rho = pressure / (r_specific * self.initial_temperature)
                mixture_list.append((species, (rho * tube_volume).to('kg')))

        return dict(mixture_list)

    def get_pressures(
            self,
            diluted=False
    ):
        """
        Cantera is used to get the mole fractions of each species, which are
        then multiplied by the initial pressure to get each partial pressure.
        """
        if diluted:
            cantera_solution = self.diluted
        else:
            cantera_solution = self.undiluted
        mixture_list = []
        for i, species in enumerate(cantera_solution.species_names):
            if cantera_solution.X[i] > 0:
                mixture_list.append((species,
                                    self.initial_pressure *
                                    cantera_solution.X[i]))
        return dict(mixture_list)
