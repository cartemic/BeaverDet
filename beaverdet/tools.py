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


# TODO: moved get_flange_limits_from_csv to tube.Tube()
# TODO: moved check_materials to tube.Tube()
# TODO: moved collect_tube_materials to tube.Tube()
# TODO: moved get_material_groups to tube.Tube()
# TODO: moved window_sympy_solver to tube.Window()
# TODO: moved import_thread_specs to tube.Bolt()
# TODO: moved get_thread_property to tube.Bolt()
# TODO: removed get_thread_tpi entirely
# TODO: moved get_pipe_stress_limits to tube.Tube()
# TODO: moved calculate_laminar_flamespeed to thermochem
# TODO: moved get_equil_sound_speed to thermochem


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
            actual_dimension_type +
            ' is not '
            + units[dimension_type]
        )

    return True
