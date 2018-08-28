import pint
import pandas as pd
import cantera as ct
from . import tools


class TestMatrix:
    def __init__(
            self,
            initial_temperature,
            initial_pressure,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mass_fraction,
            tube_volume
    ):
        # type check to make sure temperature is a pint quantity
        tools.check_pint_quantity(
            initial_temperature,
            'temperature'
        )

        # type check to make sure pressure is a pint quantity
        tools.check_pint_quantity(
            initial_pressure,
            'pressure'
        )
