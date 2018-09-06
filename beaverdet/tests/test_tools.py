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

import pytest
import pint
from .. import tools


def test_check_pint_quantity():
    """
    Tests the check_pint_quantity() function, which makes sure a variable is
    a numeric pint quantity of the correct dimensionality. It can also ensure
    that the magnitude of the quantity is > 0.

   Conditions tested:
        - bad dimension type
        - non-pint quantity
        - non-numeric pint quantity
        - negative magnitude
        - incorrect dimensionality
        - good input
    """

    # initialize unit registry and quantity for unit handling
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # Bad dimension type
    bad_dimension_type = 'walrus'
    error_str = bad_dimension_type + ' not a supported dimension type'
    with pytest.raises(ValueError, match=error_str):
        tools.check_pint_quantity(
            quant(3, ureg.degC),
            bad_dimension_type
        )

    # Non-pint quantity
    with pytest.raises(ValueError, match='Non-pint quantity'):
        tools.check_pint_quantity(
            7,
            'length'
        )

    # Non-numeric pint quantity
    with pytest.raises(ValueError, match='Non-numeric pint quantity'):
        tools.check_pint_quantity(
            quant('asdf', ureg.inch),
            'length'
        )

    # Negative magnitude
    with pytest.raises(ValueError, match='Input value < 0'):
        tools.check_pint_quantity(
            quant(-4, ureg.inch),
            'length',
            ensure_positive=True
        )

    # Incorrect dimensionality
    error_str = (
            ureg.degC.dimensionality.__str__() +
            ' is not ' +
            ureg.meter.dimensionality.__str__()
    )
    try:
        tools.check_pint_quantity(
            quant(19.2, ureg.degC),
            'length'
        )
    except ValueError as err:
        assert str(err) == error_str

    # Good input
    assert tools.check_pint_quantity(
        quant(6.3, ureg.inch),
        'length'
    )
    assert tools.check_pint_quantity(
        quant(-8, ureg.inch),
        'length'
    )
