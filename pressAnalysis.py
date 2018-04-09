# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:46:08 2017

@author: cartemic
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# input desired pipe ID and whether 'nearest' or 'minimum' is desired
requestedDiam = 6
diamType = 'nearest'
schedule = '80'
DLF = np.array([2, 4])
tolPercent = 10
shockPR = 44.23
relaxP = 1#0.4
FS = 1.5
T = np.linspace(0, 400, num=1000)

# tension vs temperature
ASME = {}
ASME['maxTens'] = 1000 * np.array([16.7, 16.7, 14.1, 12.7, 11.7, 10.9, 10.4,
                                   10.2, 10, 9.8, 9.6, 9.4, 9.2, 8.9, 8.8, 8,
                                   7.9, 6.5, 6.4])
ASME['T'] = np.array([0, 100, 200, 300, 400, 500, 600, 650, 700, 750, 800, 850,
                      900, 950, 1000, 1050, 1100, 1150, 1200])

maxTens = np.interp(T, ASME['T'], ASME['maxTens'])


# get dynamic load factor for a given pipe geometry
def getDLF(thk, OD, Dcj, plusOrMinus):
    plusOrMinus = plusOrMinus / 100
    h = thk * 0.0254
    R = np.average([OD, OD-2 * thk]) / 2 * 0.0254
    E = np.average([134, 152]) * 1e9
    rho = 7970
    nu = 0.27

    Vc0 = (
           (E**2 * h**2) /
           (3 * rho**2 * R**2 * (1-nu**2))
          )**(1/4)

    if Dcj >= (1-plusOrMinus) * Vc0 and Dcj <= (1+plusOrMinus) * Vc0:
        return(4.)
    elif Dcj < Vc0:
        return(1.)
    else:
        return(2.)


# collect thicknesses at desired diameter
def getPmax(requestedID, diamType):
    NPS = np.array([1/2, 3/4, 1, 1+1/4, 1+1/2, 2, 2+1/2, 3, 4, 5, 6, 8, 10,
                    12])
    OD = np.array([0.84, 1.05, 1.315, 1.66, 1.9, 2.375, 2.875, 3.5, 4.5, 5.563,
                   6.625, 8.625, 10.75, 12.75])
    thkList = {'40': np.array([0.109, 0.113, 0.133, 0.14, 0.145, 0.154, 0.203,
                               0.216, 0.237, 0.258, 0.280, 0.322, 0.365, 0.406
                               ]),
               '80': np.array([0.147, 0.154, 0.179, 0.191, 0.2, 0.218, 0.276,
                               0.300, 0.337, 0.375, 0.432, 0.5, 0.5, 0.5
                               ]),
               '160': np.array([0.187, 0.219, 0.25, 0.25, 0.281, 0.344, 0.375,
                                0.438, 0.531, 0.625, 0.719, 0.906, 1.125, 1.312
                                ]),
               'XXS': np.array([0.294, 0.308, 0.358, 0.382, 0.4, 0.436, 0.552,
                                0.6, 0.674, 0.75, 0.864, 0.875, 1, 1
                                ])
               }

    # determine thickness, diameter, and NPS
    t = {}
    selectedDiam = {}
    selectedNPS = {}
    if diamType == 'nearest' or diamType == 'closest':
        for k in thkList:
            # find ID closest to requested ID
            theIndex = np.abs((OD-2*thkList[k])-requestedID).argmin()
            selectedDiam[k] = OD[theIndex]
            selectedNPS[k] = NPS[theIndex]
            t[k] = thkList[k][theIndex]
    elif diamType == 'minimum':
        for k in thkList:
            # find first ID where ID >= requested ID
            theIndex = np.min([i for i in range(len(OD)) if
                               OD[i] - 2 * thkList[k][i] >= requestedID])
            selectedDiam[k] = OD[theIndex]
            selectedNPS[k] = NPS[k]
            t[k] = thkList[k][theIndex]
    else:
        t = 0
        selectedDiam = 0.01
        selectedNPS = 0


# collect thicknesses at desired diameter
def pipeSpecs(requestedID, diamType, schedule):
    NPS = np.array([1/2, 3/4, 1, 1+1/4, 1+1/2, 2, 2+1/2, 3, 4, 5, 6, 8, 10,
                    12])
    OD = np.array([0.84, 1.05, 1.315, 1.66, 1.9, 2.375, 2.875, 3.5, 4.5, 5.563,
                   6.625, 8.625, 10.75, 12.75])
    thkList = {'40': np.array([0.109, 0.113, 0.133, 0.14, 0.145, 0.154, 0.203,
                               0.216, 0.237, 0.258, 0.280, 0.322, 0.365, 0.406
                               ]),
               '80': np.array([0.147, 0.154, 0.179, 0.191, 0.2, 0.218, 0.276,
                               0.300, 0.337, 0.375, 0.432, 0.5, 0.5, 0.5
                               ]),
               '160': np.array([0.187, 0.219, 0.25, 0.25, 0.281, 0.344, 0.375,
                                0.438, 0.531, 0.625, 0.719, 0.906, 1.125, 1.312
                                ]),
               'XXS': np.array([0.294, 0.308, 0.358, 0.382, 0.4, 0.436, 0.552,
                                0.6, 0.674, 0.75, 0.864, 0.875, 1, 1
                                ])
               }

    # determine thickness, diameter, and NPS
    if diamType == 'nearest' or diamType == 'closest':
        # find ID closest to requested ID
        theIndex = np.abs((OD-2*thkList[schedule])-requestedID).argmin()
    elif diamType == 'minimum':
        # find first ID where ID >= requested ID
        theIndex = np.min([i for i in range(len(OD)) if
                           OD[i] - 2 * thkList[schedule][i] >= requestedID])
    else:
        return()

    selectedOD = OD[theIndex]
    selectedID = selectedOD - 2 * thkList[schedule][theIndex]
    selectedNPS = NPS[theIndex]
    h = thkList[schedule][theIndex]
    return{
            'OD': selectedOD,
            'ID': selectedID,
            'h': h,
            'NPS': selectedNPS
            }


# plot minimum schedule values
pDesign = np.array(range(1, 8))
for factor in DLF:
    sched = [1000 * pressure * relaxP * FS * factor * shockPR * 14.7 /
             maxTens for pressure in pDesign]
    '''
    for i in range(len(sched)):
        for j in range(len(sched[i])):
            if sched[i][j] < 80:
                sched[i][j] = 80
            elif sched[i][j] < 160:
                sched[i][j] = 160
            else:
                sched[i][j] = 0
    '''
    plt.figure()
    [plt.plot(T, sched[i], label='P0 = {} atm'
                                 .format(pDesign[i])) for i in
        range(len(sched))]
    plt.grid('on')
    plt.xlim([0, 400])
    plt.ylim([0, 180])
    plt.xlabel('T (°F)')
    plt.ylabel('Minimum Schedule')
    plt.title('DLF = {}'.format(factor))
    plt.plot([0, 400], [80, 80], 'k')
    plt.plot([0, 400], [160, 160], 'k')
    plt.legend()

# plot DLF
Dcj = np.linspace(1, 3, num=1000) * 1000
pipe = pipeSpecs(requestedDiam, diamType, schedule)
DLF = [getDLF(pipe['h'], pipe['OD'], velocity, tolPercent) for velocity in Dcj]
plt.figure()
plt.plot(Dcj, DLF)
plt.grid('on')
plt.xlabel('$D_{CJ}$ (m/s)')
plt.ylabel('DLF')
plt.title('Dynamic Load Factor (Schedule {0} NPS {1}, tolerance $\pm${2}%)'
          .format(schedule, pipe['NPS'], tolPercent))
