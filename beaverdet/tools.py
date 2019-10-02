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
# TODO: organize imports, change `pint quantity` in type notes to
#  pint.quantity._Quantity
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
    """
    TODO: add documentation
    Returns
    -------

    """
    mechanism_path = os.path.join(
        os.path.split(os.path.abspath(ct.__file__))[0],
        'data'
    )

    available = {item for item in os.listdir(mechanism_path) if
                 ('.cti' in item) or ('.xml' in item)}

    return available
