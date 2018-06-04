import cantera as ct
import pint
import numpy as np

ureg = pint.UnitRegistry()
quant = ureg.Quantity

initial_pressure = quant(1, 'atm')
initial_temperature = quant(300, 'K')

mechanism = 'gri30.cti'
fuel = 'CH4'
oxidizer = {'O2': 1, 'N2': 3.76}
equivalence = 1
gas = ct.Solution(mechanism)
gas.set_equivalence_ratio(equivalence, fuel, oxidizer)
gamma = gas.cp/gas.cv

dlf = 2
fs = 4
asme_fs = 4
max_stress = quant(15.7, 'ksi')
diameter = quant((6.625+5.761)/2, 'in')
thk = quant(0.432, 'in')

a0_m_s = 351.8239862504165
Vcj_m_s = 1812.1314269229736
M_cj_2 = (Vcj_m_s / a0_m_s)**2

P_refl = max_stress * 2 * thk * asme_fs / (diameter * fs * dlf)
# P_cj =  P_refl / (
#             5 * (4 * gamma) * gamma + 1 + np.sqrt(
#             (17 * gamma**2) + 2 * gamma + 1
#             )
#         )
P_cj = P_refl / 2.5
P_0 =  P_cj * (gamma + 1) / (gamma * M_cj_2)

print('gamma    ', gamma)
print('reflected', P_refl.to('atm'))
print('cj       ', P_cj.to('atm'))
print('initial  ', P_0.to('atm'))

