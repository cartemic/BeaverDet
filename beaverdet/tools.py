# -*- coding: utf-8 -*-
"""
PURPOSE:
    A series of accessories used in the function of the detonation design
    tools

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import pint
import cantera as ct
import os


def check_pint_quantity(
        quantity,
        dimension_type,
        ensure_positive=False
):
    """
    This function checks to make sure that a quantity is an instance of a pint
    quantity, and that it has the correct units.

    Currently supported dimension types:
        length
        area
        volume
        temperature
        pressure
        velocity

    Parameters
    ----------
    quantity : pint quantity
        Pint quantity which is to be checked for dimensionality
    dimension_type : str
        Dimensionality that quantity should have
    ensure_positive : bool
        Determines whether the magnitude of the pint quantity will be checked
        for positivity

    Returns
    -------
    True if no errors are raised

    """

    ureg = pint.UnitRegistry()
    units = {
        'length': ureg.meter.dimensionality.__str__(),
        'area': (ureg.meter**2).dimensionality.__str__(),
        'volume': (ureg.meter**3).dimensionality.__str__(),
        'temperature': ureg.degC.dimensionality.__str__(),
        'pressure': ureg.psi.dimensionality.__str__(),
        'velocity': (ureg.meter/ureg.second).dimensionality.__str__()
    }

    if dimension_type not in units:
        raise ValueError(dimension_type + ' not a supported dimension type')

    try:
        actual_dimension_type = quantity.dimensionality.__str__()
    except AttributeError:
        raise ValueError('Non-pint quantity')

    try:
        float(quantity.magnitude)
    except ValueError:
        raise ValueError('Non-numeric pint quantity')

    if ensure_positive:
        if quantity.to_base_units().magnitude < 0:
            raise ValueError('Input value < 0')

    if units[dimension_type] != actual_dimension_type:
        raise ValueError(
            actual_dimension_type.replace('[', '').replace(']', '') +
            ' is not '
            + units[dimension_type].replace('[', '').replace(']', '')
        )

    return True


def add_dataframe_row(
        dataframe,
        row
):
    """
    Adds a row to a pandas dataframe

    https://stackoverflow.com/questions/10715965/
    add-one-row-in-a-pandas-dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
    row : list or tuple or np.ndarray

    Returns
    -------

    """
    dataframe.loc[len(dataframe.index)] = row


def find_mechanisms():
    mechanism_path = os.path.join(
        os.path.split(os.path.abspath(ct.__file__))[0],
        'data'
    )

    available = {item for item in os.listdir(mechanism_path) if
                 ('.cti' in item) or ('.xml' in item)}

    return available


def diff(f_values, x_values):
    """
    estimates the derivative of some function f(x) using backward difference

    Parameters
    ----------
    f_values : list or tuple or np.array
        two concurrent function values with the last being the most recent,
         e.g. [f(x_n-1) and f(x_n)]
    x_values : list or tuple or np.array
        x values corresponding to the f_values array, e.g. [x_n-1, x_n]

    Returns
    -------
    float
        f'(x_n): derivative of the function at the most recent point
    """
    return (f_values[1] - f_values[0]) / (x_values[1] - x_values[0])


def new_guess(x_n, f_n, f_p_n):
    """
    calculates an updated guess using Newton's method

    Parameters
    ----------
    x_n
        x_n: independent variable at the current point
    f_n
        f(x_n): function evaluated at the current point
    f_p_n
        f'(x_n): derivative evaluated at the current point

    Returns
    -------
    float
        x_n+1: updated guess for the independent variable
    """
    return x_n + (f_n / f_p_n)
