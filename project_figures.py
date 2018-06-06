import beaverdet.tube_design_tools.tools as tools
import beaverdet.tube_design_tools.accessories as acc
import cantera as ct
import pint
import math
import pprint
import brewer2mpl
from matplotlib import pyplot as plt
import json
import numbers
import numpy as np
import warnings
import multiprocessing as mp
warnings.filterwarnings("ignore")


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
    a_0 = acc.get_equil_sound_speed(
        temperature=initial_temperature,
        pressure=pipe_properties['pressures']['initial'],
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
    pipe_properties['speeds']['product sound'] = a_0

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


def quantity_remover(mydict):
    """
    removes pint quantities to make json output happy

    Parameters
    ----------
    mydict

    Returns
    -------

    """
    newdict = dict()
    for key, item in mydict.items():
        try:
            item.magnitude
            newdict[key] = (item.magnitude, item.units.format_babel())
        except AttributeError:
            if isinstance(item, str) or isinstance(item, numbers.Number):
                newdict[key] = item
            else:
                newdict[key] = quantity_remover(item)
    return newdict


def quantity_putter_backer(mydict):
    """
    undoes the work of quantity_remover

    Parameters
    ----------
    mydict

    Returns
    -------

    """
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    newdict = dict()
    for key, item in mydict.items():
        if isinstance(item, list):
            newdict[key] = quant(item[0], item[1])
        elif isinstance(item, str) or isinstance(item, numbers.Number):
            newdict[key] = item
        else:
            newdict[key] = quantity_putter_backer(item)
    return newdict


def plotify(x_array,
            y_array,
            plot_title,
            xlabel,
            ylabel
):
    bg_color = '#222222'
    fg_color = '#ffffff'
    axis_font_size = 16
    title_font_size = 36
    fig = plt.figure(facecolor=bg_color, edgecolor=fg_color, figsize=(12, 6))
    axes = fig.add_subplot(111)
    # axes.set_aspect(0.1875)
    bmap = brewer2mpl.get_map('Dark2', 'Qualitative', 8)
    axes.set_color_cycle(bmap.mpl_colors)
    axes.grid(True, alpha=0.5, linestyle=':')
    axes.patch.set_facecolor(bg_color)
    axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
    axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
    axes.set_xlabel(xlabel, color=fg_color, weight='bold')
    axes.set_ylabel(ylabel, color=fg_color, weight='bold')
    axes.xaxis.label.set_size(axis_font_size*1.25)
    axes.yaxis.label.set_size(axis_font_size*1.25)
    axes.set_title(plot_title, color=fg_color, weight='bold')
    axes.title.set_size(title_font_size)
    for ctr, spine in enumerate(axes.spines.values()):
        spine.set_color(fg_color)
        if ctr % 2:
            spine.set_visible(False)
        else:
            spine.set_linewidth(2)
    for xtick, ytick in zip(axes.xaxis.get_major_ticks(), axes.yaxis.get_major_ticks()):
        xtick.label1.set_fontsize(axis_font_size)
        xtick.label1.set_fontweight('bold')
        ytick.label1.set_fontsize(axis_font_size)
        ytick.label1.set_fontweight('bold')

    styles = ['-', '--', ':', '.-']
    if len(y_array.shape) > 1:
        for idx, y_subarray in enumerate(y_array):
            plt.plot(x_array, y_subarray, styles[idx%len(styles)], axes=axes, linewidth=2)
            leg = plt.legend(['Methane', 'Hydrogen'], fontsize=axis_font_size, facecolor=None, edgecolor=None, frameon=False)
            for text in leg.get_texts():
                text.set_color(fg_color)
                text.set_weight('bold')
    else:
        plt.plot(x_array, y_array, axes=axes, linewidth=2)
    # plt.savefig(file_name+'.png', bbox='tight', facecolor=bg_color)
    return axes, fig


def run_studies():
    # constant kwargs
    pipe_material = '316L'
    desired_fs = 4
    desired_blockage_ratio = 0.45
    window_width = quant(2.5, 'in')
    window_height = quant(5.75, 'in')
    window_desired_fs = 2
    bolt_engagement_length = quant(0.5, 'inch')
    bolt_thread_size = '1/4-28'
    mechanism = 'gri30.cti'

    # varying kwargs
    different_sizes = ['1', '2', '3', '4', '5', '6']
    different_num_bolts = list(np.linspace(5, 20, 6, dtype=int))
    different_schedules = ['5', '10', '40', '80', '160', 'XXH']
    different_temps = list(np.linspace(300, 400, 6))
    different_equivs = list(np.linspace(0.75, 1.25, 6))
    different_fuels = ['CH4', 'H2']

    # variable kwargs
    pipe_schedule = '80'
    nominal_size = '6'
    num_window_bolts = 20
    initial_temperature = quant(300, 'K')
    equivalence = 1
    # fuel = 'CH4'

    # oxidizer = {'O2': 1, 'N2': 3.76}
    # gas = ct.Solution(mechanism)
    # gas.set_equivalence_ratio(
    #     equivalence,
    #     fuel,
    #     oxidizer
    # )
    # gas_mixture = gas.mole_fraction_dict()
    # print('\nstarting sizes')
    # # run size study
    # results = dict()
    # for varied_size in different_sizes:
    #     print('running ', varied)
    #     results[varied_size] = dict()
    #     for fuel in different_fuels:
    #         print('    ', fuel)
    #         # set oxidizer and get gas mixture
    #         oxidizer = {'O2': 1, 'N2': 3.76}
    #         gas = ct.Solution(mechanism)
    #         gas.set_equivalence_ratio(
    #             equivalence,
    #             fuel,
    #             oxidizer
    #         )
    #         gas_mixture = gas.mole_fraction_dict()
    #         results[varied_size][fuel] = build_pipe(
    #             pipe_schedule,
    #             varied_size,  # varying
    #             pipe_material,
    #             desired_fs,
    #             desired_blockage_ratio,
    #             window_width,
    #             window_height,
    #             window_desired_fs,
    #             num_window_bolts,
    #             bolt_engagement_length,
    #             bolt_thread_size,
    #             initial_temperature,
    #             gas_mixture,  # varying
    #             mechanism
    #         )
    # results = quantity_remover(results)
    # with open('size_study.json', 'w') as file:
    #     json.dump(results, file)
    # print('\nstarting bolts')
    # # run bolt study
    # results = dict()
    # for varied in different_num_bolts:
    #     print('running ', varied)
    #     results[str(varied)] = dict()
    #     for fuel in different_fuels:
    #         print('    ', fuel)
    #         # set oxidizer and get gas mixture
    #         oxidizer = {'O2': 1, 'N2': 3.76}
    #         gas = ct.Solution(mechanism)
    #         gas.set_equivalence_ratio(
    #             equivalence,
    #             fuel,
    #             oxidizer
    #         )
    #         gas_mixture = gas.mole_fraction_dict()
    #         results[str(varied)][fuel] = build_pipe(
    #             pipe_schedule,
    #             nominal_size,
    #             pipe_material,
    #             desired_fs,
    #             desired_blockage_ratio,
    #             window_width,
    #             window_height,
    #             window_desired_fs,
    #             int(varied),  # varying
    #             bolt_engagement_length,
    #             bolt_thread_size,
    #             initial_temperature,
    #             gas_mixture,  # varying
    #             mechanism
    #         )
    # results = quantity_remover(results)
    # pprint.pprint(results)
    # with open('bolt_study.json', 'w') as file:
    #     json.dump(results, file)
    #
    # print ('\nstarting sched')
    # # run schedule study
    # results = dict()
    # for varied in different_schedules:
    #     print('running ', varied)
    #     results[str(varied)] = dict()
    #     for fuel in different_fuels:
    #         print('    ', fuel)
    #         # set oxidizer and get gas mixture
    #         oxidizer = {'O2': 1, 'N2': 3.76}
    #         gas = ct.Solution(mechanism)
    #         gas.set_equivalence_ratio(
    #             equivalence,
    #             fuel,
    #             oxidizer
    #         )
    #         gas_mixture = gas.mole_fraction_dict()
    #         results[str(varied)][fuel] = build_pipe(
    #             varied,  # varying
    #             nominal_size,
    #             pipe_material,
    #             desired_fs,
    #             desired_blockage_ratio,
    #             window_width,
    #             window_height,
    #             window_desired_fs,
    #             num_window_bolts,
    #             bolt_engagement_length,
    #             bolt_thread_size,
    #             initial_temperature,
    #             gas_mixture,  # varying
    #             mechanism
    #         )
    # results = quantity_remover(results)
    # with open('schedule_study.json', 'w') as file:
    #     json.dump(results, file)
    #
    # print('\starting temps')
    # # run temperature study
    # results = dict()
    # for varied in different_temps:
    #     print('running ', varied)
    #     results[str(varied)] = dict()
    #     for fuel in different_fuels:
    #         print('    ', fuel)
    #         # set oxidizer and get gas mixture
    #         oxidizer = {'O2': 1, 'N2': 3.76}
    #         gas = ct.Solution(mechanism)
    #         gas.set_equivalence_ratio(
    #             equivalence,
    #             fuel,
    #             oxidizer
    #         )
    #         gas_mixture = gas.mole_fraction_dict()
    #         results[str(varied)][fuel] = build_pipe(
    #             pipe_schedule,
    #             nominal_size,
    #             pipe_material,
    #             desired_fs,
    #             desired_blockage_ratio,
    #             window_width,
    #             window_height,
    #             window_desired_fs,
    #             num_window_bolts,
    #             bolt_engagement_length,
    #             bolt_thread_size,
    #             quant(varied, 'K'),  # varying
    #             gas_mixture,  # varying
    #             mechanism
    #         )
    # results = quantity_remover(results)
    # with open('temperature_study.json', 'w') as file:
    #     json.dump(results, file)
    print('\nstarting equivs')
    # run equivalence study
    results = dict()
    for varied in different_equivs:
        print('running ', varied)
        results[str(varied)] = dict()
        for fuel in different_fuels:
            print('    ', fuel)
            # set oxidizer and get gas mixture
            oxidizer = {'O2': 1, 'N2': 3.76}
            gas = ct.Solution(mechanism)
            gas.set_equivalence_ratio(
                varied,
                fuel,
                oxidizer
            )
            gas_mixture = gas.mole_fraction_dict()
            results[str(varied)][fuel] = build_pipe(
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
                gas_mixture,  # varying
                mechanism
            )
    results = quantity_remover(results)
    with open('schedule_study.json', 'w') as file:
        json.dump(results, file)

if __name__ == '__main__':
    ureg = pint.UnitRegistry()
    quant = ureg.Quantity

    # run_studies()

    bg_color = '#222222'

    with open('schedule_study.json', 'r') as file:
        data = json.loads(file.read())

        keys = data.keys()
        methane = []
        hydrogen = []
        for key in keys:
            methane.append(data[key]['CH4']['pressures']['initial'][0])
            hydrogen.append(data[key]['H2']['pressures']['initial'][0])

        (axes, fig) = plotify(range(len(keys)-1),
                np.array([methane[:-1], hydrogen[:-1]]),
                'Initial Pressure vs. Pipe Schedule',
                'Schedule',
                'Initial Pressure (atm)')
        file_name = 'temperature_study'
        axes.xaxis.set_ticks(range(len(keys)-1))
        axes.xaxis.set_ticklabels(list(keys)[:-1])
        plt.savefig(file_name + '.png', bbox='tight', facecolor=bg_color)


    with open('size_study.json', 'r') as file:
        data = json.loads(file.read())

        keys = data.keys()
        methane = []
        hydrogen = []
        for key in keys:
            methane.append(data[key]['CH4']['pressures']['reflected'][0])
            hydrogen.append(data[key]['H2']['pressures']['reflected'][0])

        (axes, fig) = plotify([int(size) for size in data.keys()],
                              np.array([methane, hydrogen]),
                              'Initial Pressure vs. NPS',
                              'NPS (in)',
                              'Initial Pressure (atm)')
        file_name = 'size_study'
        plt.show()
        # axes.xaxis.set_ticks(range(len(keys) - 1))
        # axes.xaxis.set_ticklabels(list(keys)[:-1])

    # pipe = build_pipe(
    #     pipe_schedule,
    #     nominal_size,
    #     pipe_material,
    #     desired_fs,
    #     desired_blockage_ratio,
    #     window_width,
    #     window_height,
    #     window_desired_fs,
    #     num_window_bolts,
    #     bolt_engagement_length,
    #     bolt_thread_size,
    #     initial_temperature,
    #     gas_mixture,
    #     mechanism
    # )
    #
    # pprint.pprint(pipe, indent=4)
    #
    # with open('test_output.json', 'w') as file:
    #     json.dump(quantity_remover(pipe), file)
    #
    # with open('test_output.json', 'r') as file:
    #     print()
    #     print('loads')
    #     pprint.pprint(json.load(file))
    #     print()
    #     # print('load')
    #     # for line in file:
    #     #     pprint.pprint(json.loads(file))

    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([4, 6, 8, 8, 9])
    # y = np.array([y, x/y, y/x])
    #
    #
    # plot_title = 'Nice.'
    # xlabel = 'humdingers'
    # ylabel = 'whizzbangs'
    # file_name = 'test_plot'
    #
    # plotify(x, y, plot_title, xlabel, ylabel, file_name)

