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
# third party imports
import cantera as ct
import numpy as np
import pint

# local imports
from . import _sd, tools


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
    initial_temperature : pint.quantity._Quantity
        Initial temperature of gas mixture
    initial_pressure : pint.quantity._Quantity
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
    pint.quantity._Quantity
        Laminar flame speed in m/s
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
    flame.set_refine_criteria(
        ratio=3,
        slope=0.1,
        curve=0.1
    )
    flame.solve(loglevel=0)

    return quant(flame.u[0], 'm/s')


def get_eq_sound_speed(
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
    temperature : pint.quantity._Quantity
        Initial mixture temperature
    pressure : pint.quantity._Quantity
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
    sound_speed : pint.quantity._Quantity
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


def calculate_reflected_shock_state(
        initial_temperature,
        initial_pressure,
        species_dict,
        mechanism,
        ureg=pint.UnitRegistry(),
        use_multiprocessing=False
):
    """
    Calculates the thermodynamic and chemical state of a reflected shock
    using modified sdtoolbox functions.

    Parameters
    ----------
    initial_temperature : pint.quantity._Quantity
        Pint quantity of mixture initial temperature
    initial_pressure : pint.quantity._Quantity
        Pint quantity of mixture initial pressure
    species_dict : dict
        Dictionary of initial reactant mixture
    mechanism : str
        Mechanism to use for chemical calculations, e.g. 'gri30.cti'
    ureg : pint.UnitRegistry
        Pint unit registry
    use_multiprocessing : bool
        True to use multiprocessing for CJ state calculation, which is faster
        but requires the function to be run from __main__

    Returns
    -------
    dict
        Dictionary containing keys 'reflected' and 'cj'. Each of these
        contains 'speed', indicating the related wave speed, and 'state',
        which is a Cantera gas object at the specified state.
    """
    quant = ureg.Quantity

    # define gas objects
    initial_gas = ct.Solution(mechanism)
    reflected_gas = ct.Solution(mechanism)

    # define gas states
    initial_temperature = initial_temperature.to('K').magnitude
    initial_pressure = initial_pressure.to('Pa').magnitude

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
    cj_calcs = _sd.Detonation.cj_speed(
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
     reflected_gas] = _sd.Reflection.reflect(
        initial_gas,
        cj_calcs['cj state'],
        reflected_gas,
        cj_calcs['cj speed']
    )

    return {
        'reflected': {
            'speed': quant(
                reflected_speed,
                'm/s'
            ),
            'state': reflected_gas
        },
        'cj': {
            'speed': quant(
                cj_calcs['cj speed'],
                'm/s'),
            'state': cj_calcs['cj state']
        }
    }


class Mixture:
    """
    An object for managing initial gas mixtures. Mixtures can be undiluted or
    diluted by mole fraction.
    """
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
        """
        TODO: allow for mixed fuel, oxidizer, diluent, and update docstring
        Parameters
        ----------
        initial_pressure : pint.quantity._Quantity
            Initial reactant pressure
        initial_temperature : pint.quantity._Quantity
            Initial reactant temperature
        fuel : str
            Fuel species (e.g. `CH4`). Must be in the mechanism file.
        oxidizer : str
            Oxidizer species (e.g. `O2`). Must be in the mechanism file.
        diluent : str or None
            Oxidizer species (e.g. `N2`). Must either be None or in the
            mechanism file.
        equivalence : float
            Equivalence ratio.
        diluent_mole_fraction : float
            Mole fraction of diluent.
        mechanism : str
            Mechanism file to use in Cantera solution object.
        unit_registry : pint.registry.UnitRegistry or None
            Pint unit registry to convert units into. Passing the local unit
            registry from the parent method will allow the mixture's output
            quantities to be directly compared with the input quantities.
            Passing None will generate a new unit registry.
        """
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

        # initialize diluted and undiluted gas solution in Cantera
        self.undiluted = ct.Solution(mechanism)
        self.undiluted.TP = (
            initial_temperature.to('degK').magnitude,
            initial_pressure.to('Pa').magnitude
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
        self.initial_pressure = self._quant(
            initial_pressure.magnitude,
            initial_pressure.units.format_babel()
        ).to_base_units()
        self.initial_temperature = self._quant(
            initial_temperature.magnitude,
            initial_temperature.units.format_babel()
        ).to_base_units()
        self.diluent_mol_fraction = diluent_mole_fraction

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
        Sets the equivalence ratio of the undiluted mixture using Cantera.
        Modifies self.equivalence; no value is returned.

        Parameters
        ----------
        equivalence_ratio : float
            Equivalence ratio to set.

        Returns
        -------

        """
        equivalence_ratio = float(equivalence_ratio)

        # set the equivalence ratio
        self.undiluted.set_equivalence_ratio(equivalence_ratio,
                                             self.fuel,
                                             self.oxidizer)
        if self.diluent and self.diluent_mol_fraction:
            self.add_diluent(self.diluent, self.diluent_mol_fraction)

        self.equivalence = equivalence_ratio

    def add_diluent(
            self,
            diluent,
            mole_fraction
    ):
        """
        TODO: allow for mixed fuel, oxidizer, diluent, and update docstring
        Adds a diluent to an undiluted mixture, keeping the same equivalence
        ratio.

        Parameters
        ----------
        diluent : str
            Diluent species (e.g. `N2`). Must be in the mechanism file.
            You are not allowed to dilute with fuel or oxidizer. Mole fraction
            cannot be less than zero.
        mole_fraction : float
            Mole fraction of diluent to add.
        """
        # make sure diluent is available in mechanism and isn't the fuel or ox
        if diluent not in self.undiluted.species_names:
            raise ValueError('Bad diluent: {}'.format(diluent))
        elif diluent in [self.fuel, self.oxidizer]:
            raise ValueError('You can\'t dilute with fuel or oxidizer!')
        elif mole_fraction > 1. or mole_fraction < 0:
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

        # create cantera solution if one doesn't exist
        self.diluted = ct.Solution(self.mechanism)
        self.diluted.TPX = (
            self.initial_temperature.to('degK').magnitude,
            self.initial_pressure.to('Pa').magnitude,
            species
        )

    def get_masses(
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

        Parameters
        ----------
        tube_volume : pint.quantity._Quantity
            Total volume of detonation tube
        diluted : bool
            Tells the method which solution object to use.

        Returns
        -------
        dict
            Dictionary containing the total mass of each species in the
            reactant mixture.
        """
        tools.check_pint_quantity(
            tube_volume,
            'volume',
            ensure_positive=True
        )

        tube_volume = self._quant(
            tube_volume.magnitude,
            tube_volume.units.format_babel()
        )

        if diluted and self.diluted is None:
            raise ValueError('Mixture has not been diluted')
        elif diluted:
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

        Parameters
        ----------
        diluted : bool
            Tells the method which solution object to use.

        Returns
        -------
        dict
            Dictionary containing the partial pressure of each species in the
            reactant mixture.
        """
        if diluted and self.diluted is None:
            raise ValueError('Mixture has not been diluted')
        elif diluted:
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
