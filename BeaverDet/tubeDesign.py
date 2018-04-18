# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:50:55 2017

@author: cartemic
"""

from detTubeAnalysisTools import pipe, reflection, window, spiral, flange, boltPattern
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc

plt.close("all")

# %% INPUT

# decide whether to include plots in calls to *.info()
info_plots = False

# window bolt size
bolt_size = '1/4-28'

# set size (4 or 6)
size = 6

# main section inputs
window_dims = {}
main = {}
main['ID type'] = 'nearest'
main['FS scale'] = 1#2/3. #/1.85
if size == 6:
    main['ID'] = 6.
    window_dims['w'] = 2.5
    window_dims['l'] = 5.75
    window_dims['t'] = 1
    main['schedule'] = 80
else:
    main['ID'] = 4.
    window_dims['w'] = 3.5
    window_dims['l'] = 3.5
    window_dims['t'] = 1.25
    main['schedule'] = 160

# mixing section inputs
mix = {}
mix['schedule'] = 80
mix['ID'] = 1
mix['ID type'] = 'nearest'
mix['FS scale'] = main['FS scale'] * 2.

# mixing fan inputs
mix_fan = {}
mix_fan['schedule'] = 160
mix_fan['ID'] = 2.25
mix_fan['ID type'] = 'minimum'
mix_fan['FS scale'] = mix['FS scale']  # set to 4 due to DLF see above (ASME=4)

wave_speed = 1800  # 1400
sound_speed = 600
gamma = 1.1

# %% CALCULATION

# main pipe
main_pipe = pipe('main pipe',
                 main['ID'],
                 main['ID type'],
                 main['schedule'],
                 FS=main['FS scale'],
                 wave_speed_m_s=wave_speed)
P_max_atm = max(main_pipe.P_max_atm)

# mixing pipe
mix_pipe = pipe('mixing pipe',
                mix['ID'],
                mix['ID type'],
                mix['schedule'],
                FS=mix['FS scale'],
                wave_speed_m_s=wave_speed)

# mixing fan pipe
mix_fan_pipe = pipe('mixing fan pipe',
                    mix_fan['ID'],
                    mix_fan['ID type'],
                    mix_fan['schedule'],
                    FS=mix_fan['FS scale'],
                    wave_speed_m_s=wave_speed)

# main flanges
main_flange = flange('main flange', P_max_atm)

# DDT spiral
DDT = spiral(main_pipe.ID,
             max_pressure_difference_atm=P_max_atm,
             add_struts=False,
             number_of_struts=4,
             FS_struts=1.25)

# viewing section
VS = window(P_in_atm=P_max_atm,
            l=window_dims['l'],
            w=window_dims['w'],
            t=window_dims['t'])

# reflection section
refl = reflection(main_pipe.P_max_atm,
                  a0_m_s=sound_speed,
                  Vcj_m_s=wave_speed,
                  gamma=gamma)

# print warning about DLF
if main_pipe.design_DLF < 2:
    print('''WARNING: DLF is less than 2. If your input velocity was less than
         C-J, you should probably use a DLF of 4!!''')

# window bolts (non-failure)
F = window_dims['l'] * window_dims['w'] * main_pipe.P_max
strong_bolts = boltPattern(N_bolts=20*np.ones(main_pipe.T.shape),
                           bolt_size=bolt_size,
                           temp_F=main_pipe.T,
                           total_force_lbf=F)

# window bolts (failure) initial
weak_bolts = boltPattern(N_bolts=13*np.ones(main_pipe.T.shape),
                         bolt_size=bolt_size,
                         temp_F=main_pipe.T,
                         total_force_lbf=F,
                         bolt_max_stress_psi=50000,
                         bolt_temp_derating='brass')

# correct bolts to get closest to ideal failure limit
N_bolts_adjusted = 13 / weak_bolts.bolt['FS']

# Recalculate window failure with ideal number of bolts
weak_bolts = boltPattern(np.round(N_bolts_adjusted),
                         bolt_size=bolt_size,
                         temp_F=main_pipe.T,
                         total_force_lbf=F,
                         bolt_max_stress_psi=50000,
                         bolt_temp_derating='brass')

# %% OUTPUT

# print information about components
main_pipe.info(plot=info_plots)
main_flange.info(plot=info_plots)
mix_pipe.info(plot=info_plots)
mix_fan_pipe.info(plot=info_plots)
DDT.info(plot=info_plots)
VS.info(plot=info_plots)
refl.info(plot=info_plots)

# plot initial pressure limits
plt.figure('Allowable Initial Pressure vs. Pipe Temperature')
plt.plot(main_pipe.T, refl.P0_atm, label='Main Pipe')
plt.grid('on')
plt.xlim([min(main_pipe.T), max(main_pipe.T)])
plt.xlabel('Pipe Temperature (°F)')
plt.ylabel('Max Initial Pressure (atm)')
plt.title('''Pressure Envelope (Schedule {2}, NPS {3})
Pipe FS = {0: 1.1f}, DLF = {1}, TOTAL = {4: 1.1f}'''
          .format(4*main['FS scale'],
                  main_pipe.design_DLF,
                  main_pipe.schedule,
                  main_pipe.NPS,
                  main_pipe.design_DLF * 4*main['FS scale']))
plt.tight_layout()

# plot max allowable pressure for all pipes
plt.figure('Maximum Allowable Pressure vs. Pipe Temperature')
plt.plot(main_pipe.T, main_pipe.P_max_atm, label='Main Pipe')
plt.plot(mix_pipe.T, mix_pipe.P_max_atm, label='Mixing Pipe')
plt.plot(mix_fan_pipe.T, mix_fan_pipe.P_max_atm, label='Mixing Fan Pipe')
plt.xlim([np.min([main_pipe.T, mix_pipe.T, mix_fan_pipe.T]),
          np.max([main_pipe.T, mix_pipe.T, mix_fan_pipe.T])])
plt.xlabel('Pipe Temperature (°F)')
plt.ylabel('Max Allowable Pressure (atm)')
plt.title('Maximum Allowable Pressure vs. Pipe Temperature\n' +
          '(Includes ASME Safety Factor of 4)')
plt.grid('on')
plt.legend()

# plot strong bolt FS
plt.figure('Window Retention Safety Factors')
plt.plot(main_pipe.T, strong_bolts.bolt['FS'], 'k:', label='Mount Bolts')
plt.plot(main_pipe.T, strong_bolts.plate['FS'], 'k', label='Mount Plate')
plt.plot(main_pipe.T, weak_bolts.bolt['FS'], 'r:', label='Blowoff Bolts')
plt.plot(main_pipe.T, weak_bolts.plate['FS'], 'r', label='Blowoff Plate')
plt.xlim([0, 500])
plt.xlabel('Temperature (°F)')
plt.ylabel('Safety Factor')
plt.title('Window Retention Safety Factors')
plt.grid('on')
plt.legend()

# plot bolt count vs temp
plt.figure('Ideal Bolt Count (Blowoff)')
plt.plot(main_pipe.T, N_bolts_adjusted)
plt.xlim([0, 500])
plt.xlabel('Temperature (°F)')
plt.ylabel('Safety Factor')
plt.title('Ideal Bolt Count (Blowoff)')
plt.grid('on')

# vary CJ mach number and plot results
a_0 = 600
M_cj = np.array(range(8, 21, 2))/2
plt.figure('Initial Pressure and Temperature Limits')
ax = plt.subplot(111)

for i in range(len(M_cj)):
    refl_2 = reflection(main_pipe.P_max_atm, a0_m_s=a_0, Vcj_m_s=a_0*M_cj[i])
    ax.plot(main_pipe.T, refl_2.P0_atm, label=M_cj[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.show()
plt.grid('on')
plt.xlim([0, 450])
plt.xlabel('Pipe Temperature (°F)')
plt.ylabel('Max Initial Pressure (atm)')
ax.legend(title='$M_{C-J}$', loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('''Initial P-T Limits With Varying C-J Mach for $\gamma$ = {0}'''
          .format(refl_2.gamma,
                  main_pipe.design_DLF,
                  main_pipe.schedule,
                  main_pipe.NPS,
                  main_pipe.design_DLF * 4*main['FS scale']))

# vary gamma and plot results
a_0 = 600
V_cj = 10 * a_0
gamma = np.linspace(1, 1.6, 5)
plt.figure('Initial Pressure and Temperature Limits 2')
ax = plt.subplot(111)

for i in range(len(gamma)):
    refl_2 = reflection(main_pipe.P_max_atm, a0_m_s=a_0, Vcj_m_s=V_cj,
                        gamma=gamma[i])
    ax.plot(main_pipe.T, refl_2.P0_atm, label=gamma[i])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.show()
plt.grid('on')
plt.xlim([0, 450])
plt.xlabel('Pipe Temperature (°F)')
plt.ylabel('Max Initial Pressure (atm)')
ax.legend(title='$\gamma = C_p/C_v$', loc='center left',
          bbox_to_anchor=(1, 0.5))
plt.title('''Initial P-T Limits With Varying $\gamma$
for C-J Mach Number of {0}'''
          .format(V_cj / a_0,
                  main_pipe.design_DLF,
                  main_pipe.schedule,
                  main_pipe.NPS,
                  main_pipe.design_DLF * 4*main['FS scale']))
