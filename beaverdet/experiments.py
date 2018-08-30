import pint
import pandas as pd
import cantera as ct
from . import tools, thermochem


class TestMatrix:
    def __init__(
            self,
            initial_temperature,
            initial_pressure,
            fuel,
            oxidizer,
            diluent,
            equivalence,
            num_replicates,
            diluent_mole_fraction,
            tube_volume
    ):
        tools.check_pint_quantity(
            initial_temperature,
            'temperature'
        )
        tools.check_pint_quantity(
            initial_pressure,
            'pressure'
        )
        tools.check_pint_quantity(
            tube_volume,
            'volume'
        )

        self.mixture = thermochem.Mixture(
            initial_pressure,
            initial_temperature,
            fuel,
            oxidizer,
            diluent=diluent
        )
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.diluent = diluent
        self.tube_volume = tube_volume
        self.base_replicate = None
        self.replicates = [None for item in range(num_replicates)] # initialize dataframe

        # ensure that equivalence is iterable and numeric
        try:
            len(equivalence)
        except TypeError:
            # equivalence is not iterable. ensure that it is numeric and then
            #  make it iterable
            try:
                equivalence / 7
                equivalence = [equivalence]
            except TypeError:
                # neither iterable nor numeric
                raise TypeError('equivalence is neither iterable nor numeric')
        try:
            # check that all items are numeric
            [item / 7 for item in equivalence]
        except TypeError:
            raise TypeError('equivalence has non-numeric items')

        self.equivalence = equivalence

        # repeat for diluent mole fraction
        try:
            len(diluent_mole_fraction)
        except TypeError:
            # equivalence is not iterable. ensure that it is numeric and then
            #  make it iterable
            try:
                diluent_mole_fraction / 7
                diluent_mole_fraction = [diluent_mole_fraction]
            except TypeError:
                # neither iterable nor numeric
                raise TypeError('diluent_mole_fraction is neither iterable' +
                                ' nor numeric')
        try:
            # check that all items are numeric
            [item / 7 for item in diluent_mole_fraction]
        except TypeError:
            raise TypeError('diluent_mole_fraction has non-numeric items')

        self.diluent_mole_fraction = diluent_mole_fraction

    def build_replicate(
            self
    ):
        # initialize dataframe

        for phi in self.equivalence:
            # set mixture undiluted equivalence ratio
            if self.diluent:
                for dil_mf in self.diluent_mole_fraction:
                    # dilute mixture
                    # add row to dataframe
                    pass
            else:
                # add row to dataframe
                pass

        # save dataframe to self

    def randomize_replicate(
            self
    ):
        pass
