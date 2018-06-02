import beaverdet.tube_design_tools.tools as tools
import beaverdet.tube_design_tools.accessories as acc
import cantera as ct
import pint
import math
import pprint
import brewer2mpl
from matplotlib import pyplot as plt


def build_pipe(
        pipe_schedule,
        nominal_size,
        pipe_material,
        desired_fs,
        desired_blockage_ratio,
        window_width,
        window_height,
        window_desired_fs,
        num_window_bolts,
        bolt_engagement_length,
        bolt_thread_size,
        initial_temperature,
        gas_mixture,
        mechanism
):
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    pipe_properties = dict()
    pipe_properties['schedule'] = pipe_schedule
    pipe_properties['size'] = 'NPS ' + nominal_size
    pipe_properties['material'] = pipe_material
    pipe_properties['safety factor'] = desired_fs
    pipe_properties['temperatures'] = dict()
    pipe_properties['temperatures']['initial'] = initial_temperature

    pipe_dimensions = acc.get_pipe_dimensions(
        pipe_schedule,
        nominal_size
    )

    # get blockage diameter, find next closest 1/8 inch, and recalculate with
    # closest available size
    pipe_dimensions['blockage diameter'] = tools.calculate_spiral_diameter(
        pipe_dimensions['inner diameter'],
        desired_blockage_ratio
    )
    pipe_dimensions['blockage diameter'] = round(
        pipe_dimensions['blockage diameter'].to('in') * 8
    ) / 8.
    pipe_dimensions['blockage ratio'] = tools.calculate_blockage_ratio(
        pipe_dimensions['inner diameter'],
        pipe_dimensions['blockage diameter']
    )

    # calculate initial pressure
    pipe_properties['pressures'] = dict()
    pipe_properties['pressures']['initial'] = tools.calculate_max_initial_pressure(
        pipe_material=pipe_material,
        pipe_schedule=pipe_schedule,
        pipe_nps=nominal_size,
        welded=False,
        desired_fs=desired_fs,
        initial_temperature=initial_temperature,
        species_dict=gas_mixture,
        mechanism=mechanism
    )

    # calculate DDT runup distance
    pipe_dimensions['DDT runup'] = tools.calculate_ddt_run_up(
        blockage_ratio=pipe_dimensions['blockage ratio'],
        tube_diameter=pipe_dimensions['inner diameter'],
        initial_temperature=initial_temperature,
        initial_pressure=pipe_properties['pressures']['initial'],
        species_dict=gas_mixture,
        mechanism=mechanism
    )
    pipe_properties['dimensions'] = pipe_dimensions

    # calculate cj and reflected states
    states = tools.calculate_reflected_shock_state(
        initial_pressure=pipe_properties['pressures']['initial'],
        initial_temperature=initial_temperature,
        species_dict=gas_mixture,
        mechanism=mechanism
    )
    pipe_properties['pressures']['cj'] = quant(
        states['cj']['state'].P,
        'Pa'
    ).to('atm')
    pipe_properties['pressures']['reflected'] = quant(
        states['reflected']['state'].P,
        'Pa'
    ).to('atm')
    pipe_properties['temperatures']['cj'] = quant(
        states['cj']['state'].T,
        'K'
    )
    pipe_properties['temperatures']['reflected'] = quant(
        states['reflected']['state'].T,
        'K'
    )
    pipe_properties['speeds'] = dict()
    pipe_properties['speeds']['cj'] = states['cj']['speed']
    pipe_properties['speeds']['reflected'] = states['reflected']['speed']

    # collect dynamic load factor
    pipe_properties['dynamic load factor'] = tools.get_pipe_dlf(
        pipe_material=pipe_material,
        pipe_schedule=pipe_schedule,
        nominal_pipe_size=nominal_size,
        cj_speed=pipe_properties['speeds']['cj']
    )

    # size flange
    pipe_properties['flange class'] = tools.lookup_flange_class(
        temperature=initial_temperature,
        pressure=pipe_properties['pressures']['reflected'],
        desired_material=pipe_material
    )

    # calculate window thickness, adjust to next highest 1/8 inch, and
    # recalculate
    window_rupture_modulus = quant(5950, 'psi')
    pipe_properties['window'] = dict()
    pipe_properties['window']['width'] = window_width
    pipe_properties['window']['height'] = window_height
    pipe_properties['window']['thickness'] = tools.calculate_window_thk(
        length=window_height,
        width=window_width,
        safety_factor=window_desired_fs,
        pressure=pipe_properties['pressures']['cj'],
        rupture_modulus=window_rupture_modulus
    )
    pipe_properties['window']['thickness'] = quant(
        math.ceil(
            pipe_properties['window']['thickness'].to('in').magnitude * 8
        ) / 8.,
        pipe_properties['window']['thickness'].units.format_babel()
    )
    pipe_properties['window']['safety factor'] = tools.calculate_window_sf(
        length=window_height,
        width=window_width,
        thickness=pipe_properties['window']['thickness'],
        pressure=pipe_properties['pressures']['cj'],
        rupture_modulus=window_rupture_modulus
    )

    # bolt calculations
    pipe_properties['viewing section'] = dict()
    pipe_properties['viewing section']['number of bolts'] = num_window_bolts
    pipe_properties['viewing section']['safety factors'] = tools.calculate_window_bolt_sf(
        max_pressure=pipe_properties['pressures']['reflected'],
        window_area=window_width*window_height,
        num_bolts=num_window_bolts,
        thread_size=bolt_thread_size,
        thread_class='2',
        bolt_max_tensile=quant(120, 'ksi'),
        plate_max_tensile=quant(30, 'ksi'),
        engagement_length=bolt_engagement_length
    )

    return pipe_properties


if __name__ == '__main__':
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    pipe_schedule = '80'
    nominal_size = '6'
    pipe_material = '316L'
    desired_fs = 4
    desired_blockage_ratio = 0.45
    window_width = quant(2.5, 'in')
    window_height = quant(5.75, 'in')
    window_desired_fs = 2
    num_window_bolts = 20
    bolt_engagement_length = quant(0.5, 'inch')
    bolt_thread_size = '1/4-28'
    initial_temperature = quant(300, 'K')
    mechanism = 'gri30.cti'

    fuel = 'CH4'
    oxidizer = {'O2': 1, 'N2': 3.76}
    equivalence = 1
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(
        equivalence,
        fuel,
        oxidizer
    )
    gas_mixture = gas.mole_fraction_dict()

    pipe = build_pipe(
        pipe_schedule,
        nominal_size,
        pipe_material,
        desired_fs,
        desired_blockage_ratio,
        window_width,
        window_height,
        window_desired_fs,
        num_window_bolts,
        bolt_engagement_length,
        bolt_thread_size,
        initial_temperature,
        gas_mixture,
        mechanism
    )

    pprint.pprint(pipe, indent=4)