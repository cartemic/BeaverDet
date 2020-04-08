# -*- coding: utf-8 -*-
"""
PURPOSE:
    Unit tests for accessories.py

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import pandas as pd
import pint
import pytest

from .. import tools


def test_add_dataframe_row():
    column_names = ['a', 's', 'd', 'f']
    added_row = [0, 'f', 3, 'asd']

    good_dataframe = pd.DataFrame(
        columns=column_names,
        data=[added_row],
        dtype=object
    )

    test_dataframe = pd.DataFrame(
        columns=column_names,
        dtype=object
    )
    tools.add_dataframe_row(test_dataframe, added_row)

    assert test_dataframe.equals(good_dataframe)


class TestCheckPintQuantity:
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    def test_good_input(self):
        lengths = [6.3, -8]
        for length in lengths:
            assert tools.check_pint_quantity(
                self.quant(length, 'inch'),
                'length'
            )

    def test_bad_dimension_type(self):
        bad_dimension_type = 'walrus'
        error_str = bad_dimension_type + ' not a supported dimension type'
        with pytest.raises(
                ValueError,
                match=error_str
        ):
            tools.check_pint_quantity(
                self.quant(3, 'degC'),
                bad_dimension_type
            )

    @staticmethod
    def test_non_pint_quantity():
        with pytest.raises(
                ValueError,
                match='Non-pint quantity'
        ):
            tools.check_pint_quantity(
                7,
                'length'
            )

    def test_non_numeric_quantity(self):
        with pytest.raises(
                ValueError,
                match='Non-numeric pint quantity'
        ):
            tools.check_pint_quantity(
                self.quant('asdf', 'inch'),
                'length'
            )

    def test_ensure_positive_with_negative_magnitude(self):
        with pytest.raises(
                ValueError,
                match='Input value < 0'
        ):
            tools.check_pint_quantity(
                self.quant(-4, 'in'),
                'length',
                ensure_positive=True
            )

    def test_incorrect_dimensionality(self):
        error_str = (
                self.ureg.degC.dimensionality.__str__().strip('[]') +
                ' is not ' +
                self.ureg.meter.dimensionality.__str__().strip('[]')
        )
        with pytest.raises(
            ValueError,
            match=error_str
        ):
            tools.check_pint_quantity(
                self.quant(19.2, 'degC'),
                'length'
            )
