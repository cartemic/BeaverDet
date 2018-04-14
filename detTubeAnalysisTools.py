# -*- coding: utf-8 -*-
"""
PURPOSE:
    A series of tools to aid in the design of a detonation tube.

WHODUNIT:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import sympy as sym
import pandas as pd
from os.path import exists


# define pipe class
class pipe:
    """
    WORDS FROM A WISE DETONATION MASTER:
        This is my pipe.
        There are many like it, but this one is mine.
        Without me, my pipe is nothing.
        Without my pipe, I am nothing.

    THIS IS THE pipe() FUNCTION:
        It defines a pipe as outlined below

    INPUTS:
        pipeName:   A string of your pipe's name. You wouldn't play with an
                    unnamed pipe, would you?

        ID_request: The inner diameter of pipe you would like (inches)

        ID_type:    A string of how you would like to select your ID.
                        Options are:
                            'closest':  For when you just want it close
                            'nearest':  Because I sometimes forget that the
                                        correct option is 'closest'
                            'minimum':  For when something just HAS to fit
                                        inside your tube

        schedule:   A string containing the pipe schedule your heart desires.
                        Options are:
                            '40':       weenie pipe
                            '80':       slightly more gooder pipe
                            '160':      stronk pipe
                            'XXS':      unobtainable pipe

        WHAT IT DO:
            - Finds the correct NPS size and corresponding dimensions
            - Calculates dynamic load factor for a range of CJ velocities
            - Calculates maximum pressure for a safety factor that isn't listed
              on the inputs because I'm not done making this yet

        BONUS ACTIONS:
            info:               Outputs pipe information to the console; plots
                                dynamic load factor (DLF), max allowable
                                pressure, and max initial pressure

            DLF_plus_or_minus:  Mess with this if you want to change the bounds
                                inside of which are considered to be
                                'approximately the critical velocity'
                                (input a number of percentage points)
                                DEFAULT: 10
    """

    # initialize pipe counter
    numPipes = 0

    def __init__(self,
                 pipeName, ID_request, ID_type, schedule, FS=1.,
                 wave_speed_m_s=2500, **kwargs):

        # check for DLF_plus_or_minus input
        # This is the variation (in %) from critical velocity that counts as
        # 'approximately critical velocity' for dynamic load factor
        # calculation. If no DLF_plus_or_minus is requested, use a default
        # value of 10%
        if 'DLF_plus_or_minus' in kwargs:
            self.DLF_plus_or_minus = kwargs['DLF_plus_or_minus']
        else:
            self.DLF_plus_or_minus = 10.

        # Max stress vs. temperature tabulated values
        # From ASME B31.1-2007
        self.Stress_max_psi = 1000 * np.array([16.7, 16.7, 14.1, 12.7, 11.7,
                                               10.9, 10.4, 10.2, 10, 9.8, 9.6,
                                               9.4, 9.2, 8.9, 8.8, 8, 7.9, 6.5,
                                               6.4])
        self.T = np.array([0, 100, 200, 300, 400, 500, 600, 650, 700, 750, 800,
                           850, 900, 950, 1000, 1050, 1100, 1150, 1200])

        # given information
        self.name = pipeName
        self.ID_request = float(ID_request)
        self.ID_type = ID_type
        self.schedule = str(schedule)
        self.wave_speed_m_s = wave_speed_m_s
        self.FS = FS

        # calculate actual pipe ID, OD, thickness, and NPS
        self.getPipe()

        # calculate dynamic load factor
        self.getDLF()

        # calculate max allowable pressure
        self.getPmax()

        # increment pipe counter
        pipe.numPipes += 1
        self.pipeNumber = pipe.numPipes

    def getPipe(self):
        """Finds the pipe specifications as requested by the user"""

        # Arrays of NPS sizes and their corresponding outer diameters
        NPS = np.array([
                        1/2, 3/4, 1, 1+1/4, 1+1/2, 2, 2+1/2, 3, 4, 5, 6, 8, 10,
                        12
                        ])
        OD = np.array([
                       0.84, 1.05, 1.315, 1.66, 1.9, 2.375, 2.875, 3.5, 4.5,
                       5.563, 6.625, 8.625, 10.75, 12.75
                       ])

        # dictionary of wall thicknesses for various schedules of pipe
        # corresponding to the NPS and OD arrays defined above
        thkList = {'40': np.array([
                                   0.109, 0.113, 0.133, 0.14, 0.145, 0.154,
                                   0.203, 0.216, 0.237, 0.258, 0.280, 0.322,
                                   0.365, 0.406
                                   ]),
                   '80': np.array([
                                   0.147, 0.154, 0.179, 0.191, 0.2, 0.218,
                                   0.276, 0.300, 0.337, 0.375, 0.432, 0.5, 0.5,
                                   0.5
                                   ]),
                   '160': np.array([
                                    0.187, 0.219, 0.25, 0.25, 0.281, 0.344,
                                    0.375, 0.438, 0.531, 0.625, 0.719, 0.906,
                                    1.125, 1.312
                                    ]),
                   'XXS': np.array([
                                    0.294, 0.308, 0.358, 0.382, 0.4, 0.436,
                                    0.552, 0.6, 0.674, 0.75, 0.864, 0.875, 1, 1
                                    ])
                   }

        # determine thickness, diameter, and NPS
        if self.ID_type == 'nearest' or self.ID_type == 'closest':
            # find ID closest to requested ID
            theIndex = np.abs((OD-2*thkList[self.schedule])-self.ID_request)\
                .argmin()
        elif self.ID_type == 'minimum':
            # find first ID where ID >= requested ID
            theIndex = np.min([i for i in range(len(OD)) if
                               OD[i] - 2 * thkList[self.schedule][i] >=
                               self.ID_request])
        else:
            print()
            print('FAIL! You put in a bogus ID_type! You fool!!')
            theIndex = np.NaN

        try:
            # check to see if the index where information is located is an
            # integer. If it is not, this block will fail and the user will
            # be yelled at.
            int(theIndex)

            # if integer check is passed, define pipe parameters correctly
            selectedOD = OD[theIndex]
            selectedID = selectedOD - 2 * thkList[self.schedule][theIndex]
            selectedNPS = NPS[theIndex]
            selectedThk = thkList[self.schedule][theIndex]

            # format NPS so it looks all pretty
            if selectedNPS % 1 == 0:
                selectedNPS = '{0}'.format(int(selectedNPS))
            else:
                selectedNPS = '{0}-{1}'.format(int(selectedNPS),
                                               Fraction(selectedNPS % 1))
            # return pipe specifications
            self.OD = selectedOD
            self.ID = selectedID
            self.thk = selectedThk
            self.NPS = selectedNPS

        except:
            print()
            print('Now you have no pipe specs. I bet you feel awesome.')

    def getDLF(self):
        """
        Calculates dynamic load factor as a function of C-J velocity per:

        J. Shepherd, "Structural Response of Piping to Internal Gas
        Detonation", Journal of Pressure Vessel Technology, vol. 131, issue 3,
        pp. 031204, 2009
        """
        try:
            # check to make sure there is a thickness
            self.thk

            # set limits for 'approximately Vcrit'
            bounds = self.DLF_plus_or_minus / 100

            # Convert relevant geometry to meters
            h = self.thk * 0.0254
            R = np.average([self.OD, self.OD-2 * self.thk]) / 2 * 0.0254

            # material properties for 316 SS (SI units)
            E = np.average([134, 152]) * 1e9        # elastic modulus, Pa
            rho = 7970                              # density, kg/m^3
            nu = 0.27                               # Poisson's ratio

            # calculate critical velocity
            Vc0 = (
                   (E**2 * h**2) /
                   (3 * rho**2 * R**2 * (1-nu**2))
                  )**(1/4)
            self.V_crit = Vc0

            Dcj = np.linspace(1000, 2500, num=1000)
            DLF = np.ones(Dcj.shape)
            DLF[np.logical_and((Dcj >= (1-bounds) * Vc0),
                               (Dcj <= (1+bounds) * Vc0))] = 4.
            DLF[Dcj > (1+bounds) * Vc0] = 2.
            self.DLF = {'Dcj': Dcj, 'factor': DLF}
        except:
            # Yell at the user for hosing stuff up
            print()
            print('Looks like you don''t get a DLF. Try harder next time.')

    def getPmax(self):
        """
        Calculates the pipe's maximum allowable pressure per ASME B31.1-2007
        """
        try:
            # make sure the OD exists
            self.OD

            # get dynamic load factor
            self.design_DLF = np.interp(self.wave_speed_m_s,
                                        self.DLF['Dcj'],
                                        self.DLF['factor'])

            # calculate max allowable pressure
            self.P_max = self.Stress_max_psi * 2 * self.thk /\
                (self.OD * self.FS * self.design_DLF)
            self.P_max_atm = self.P_max / 14.7
        except:
            print('You can''t have maximum pressure without an OD...')

    def info(self, plot=False):
        """
        Returns information about the pipe
        """
        try:
            # check to make sure the pipe even has an OD
            self.OD

            # print specifications
            print()
            print('*' * (len(self.name) + 4))
            print('* {} *'.format(self.name.upper()))
            print('*' * (len(self.name) + 4))
            print('NPS:           {}'.format(self.NPS))
            print('Schedule:      {}'.format(self.schedule))
            print('Safety Factor: {}'.format(self.FS*4.))
            print('ID:            {} in'.format(round(self.ID, 3)))
            print('OD:            {} in'.format(round(self.OD, 3)))
            print('thickness:     {} in'.format(round(self.thk, 3)))

            if plot:
                # show DLF
                nameStr = '{} Dynamic Load Factor'.format(self.name.title())
                plt.figure(nameStr)
                plt.clf()
                plt.plot(self.DLF['Dcj'], self.DLF['factor'])
                plt.grid('on')
                plt.xlim([min(self.DLF['Dcj']), max(self.DLF['Dcj'])])
                plt.title(nameStr)
                plt.xlabel('Wave Velocity (m/s)')
                plt.ylabel('Dynamic Load Factor (-)')

                # show maximum pressure
                nameStr = '{0} Maximum Pressure'.format(self.name.title())
                plt.figure(nameStr)
                plt.clf()
                plt.plot(self.T, self.P_max_atm)
                nameStr = '{0}\nFactor of Safety = {1}, DLF = {2}'\
                    .format(nameStr, self.FS, self.design_DLF)
                plt.title(nameStr)
                plt.grid('on')
                plt.xlim([min(self.T), max(self.T)])
                plt.xlabel('Pipe Temperature (°F)')
                plt.ylabel('Max Allowable Pressure (atm)')

        except:
            print()
            print('No information for you! Better inputs next time.')


class window:
    """
    Defines a window to be used for optical access into a detonation tube!!

    Silicon Dioxide properties from:
    https://www.crystran.co.uk/optical-materials/quartz-crystal-sio2

    Formulas:
    https://www.crystran.co.uk/userfiles/files/design-of-pressure-windows.pdf
    http://www.advancedglass.net/pdfdocs/PressureWindows.pdf
    """
    def __init__(self,
                 circular_or_rectangular='rectangular',
                 clamped_or_unclamped='clamped',
                 P_in_atm=1,
                 rupture_modulus_in_psi=5950,
                 **kwargs):
        # store keyword arguments
        self.inputs = kwargs

        self.retention = clamped_or_unclamped
        self.shape = circular_or_rectangular

        # figure out which keyword args are correct and store them as values
        goodList = [key for key in kwargs if key in ['r', 'SF', 'l', 'w', 't']]
        [setattr(self, key, kwargs[key]) for key in goodList]

        # convert pressure to psi
        self.P = 14.7 * P_in_atm

        # rupture modulus of SIO2 (psi)
        self.M = rupture_modulus_in_psi

        # get K factor
        if clamped_or_unclamped == 'clamped':
            self.K = 0.75
        elif clamped_or_unclamped == 'unclamped':
            self.K = 1.125
        else:
            self.K = np.nan
            print()
            print('ERROR: your window clamping is jacked up.')

        # check window shape and send to appropriate solver
        if circular_or_rectangular == 'rectangular':
            # check for correct kwargs
            num_correct = sum([key in kwargs for key in ['l', 'w', 'SF', 't']])
            if num_correct == 3:
                # send to rectangular window solver
                self.rectangular_window()
            elif num_correct < 3:
                print()
                print('Hey! You didn''t send enough of the right information!')
            else:
                print()
                print('DON''T OVERCONSTRAIN ME, BRO!')
            print()
        elif circular_or_rectangular == 'circular':
            # check for correct kwargs
            num_correct = sum([key in kwargs for key in ['r', 'SF', 't']])
            if num_correct == 2:
                # send to circular window solver
                self.circular_window()
            elif num_correct < 2:
                print()
                print('Hey! You didn''t send enough of the right information!')
            else:
                print()
                print('DON''T OVERCONSTRAIN ME, BRO!')
        else:
            print()
            print('ERROR: what kind of geometry did you send me? I''m broken.')

    def circular_window(self, reset=False, constraint_to_remove='t'):
        """
        For when you gots to have a circle
        """
        # set list of attributes
        good_things = ['r', 'SF', 'l', 'w', 't']

        # decide if attributes should be reset to initial state
        if reset:
            [setattr(self, key, self.inputs[key]) for key in
             good_things if key in self.inputs]

        # make sure constraint to remove is good, otherwise use 't'
        if constraint_to_remove not in good_things:
            print("ERROR: '" + constraint_to_remove +
                  "' not in list of good constraints:")
            print(good_things)
            print('Removing ''t'' instead.')
            constraint_to_remove = 't'

        # set current shape
        self.shape = 'circular'

        # define symbols
        t, r, P, K, SF, M = sym.symbols('t r P K SF M')

        # define expression
        expr = r * sym.sqrt((P * K * SF / M)) - t

        # substitute known values into expression
        expr = expr.subs([
                          ('K', self.K),
                          ('M', self.M),
                          ('P', self.P)
                          ])

        # fix all but one of the free parameters and solve
        theKeys = ['r', 'SF', 't']
        theVars = [key for key in self.inputs if key in theKeys]
        solvedFor = [key for key in theKeys if key not in theVars]
        if solvedFor == []:
            # remove a constraint
            setattr(self, constraint_to_remove, None)
            solvedFor = constraint_to_remove
            theVars.remove(constraint_to_remove)
        expr = expr.subs([
                          (theVars[i], self.inputs[theVars[i]])
                          for i in range(len(theVars))])
        try:
            exec('self.{0} = {1}'.format(solvedFor[0], sym.solve(expr)[0]))
        except Exception as err:
            if type(err).__name__ == 'NameError' and 'I' in str(err):
                print('Oops! You did something that resulted in an imaginary.')
            else:
                print('Uh-oh. Something in the .circular_window() broke')
                print(Exception)

    def rectangular_window(self, reset=False, constraint_to_remove='t'):
        """
        For when you need a rectangular window
        """
        # set list of attributes
        good_things = ['r', 'SF', 'l', 'w', 't']

        # decide if attributes should be reset to initial state
        if reset:
            [setattr(self, key, self.inputs[key]) for key in
             good_things if key in self.inputs]

        # make sure constraint to remove is good, otherwise use 't'
        if constraint_to_remove not in good_things:
            print("ERROR: '" + constraint_to_remove +
                  "' not in list of good constraints:")
            print(good_things)
            print('Removing ''t'' instead.')
            constraint_to_remove = 't'

        # set current shape
        self.shape = 'rectangular'

        # define symbols
        t, l, w, P, K, SF, M = sym.symbols('t l w P K SF M')

        # define expression
        expr = l * w * sym.sqrt((P * K * SF / (2 * M * (l**2 + w**2)))) - t

        # substitute known values into expression
        expr = expr.subs([
                          ('K', self.K),
                          ('M', self.M),
                          ('P', self.P)
                          ])

        # fix all but one of the free parameters and solve
        theKeys = ['l', 'w', 'SF', 't']
        theVars = [key for key in self.inputs if key in theKeys]
        solvedFor = [key for key in theKeys if key not in theVars]
        if solvedFor == []:
            # remove a constraint
            setattr(self, constraint_to_remove, None)
            solvedFor = constraint_to_remove
            theVars.remove(constraint_to_remove)
        expr = expr.subs([
                          (theVars[i], self.inputs[theVars[i]])
                          for i in range(len(theVars))])
        try:
            exec('self.{0} = {1}'.format(solvedFor[0], sym.solve(expr)[0]))
        except Exception as err:
            if type(err).__name__ == 'NameError' and 'I' in str(err):
                print('Oops! You did something that resulted in an imaginary.')
            else:
                print('Uh-oh. Something in the .rectangular_window() broke')

    def info(self, plot=False):
        print()
        print('******************')
        print('* VIEWING WINDOW *')
        print('******************')
        print('Shape:         {0}'.format(self.shape))
        print('Retention:     {0}'.format(self.retention))
        print('Safety Factor: {0}'.format(round(self.SF, 2)))
        if plot:
            print()
            print('- NO PLOT FOR THIS COMPONENT -')
        if self.shape == 'circular':
            print('Radius:        {0} in.'.format(round(self.r, 3)))
        else:
            print('Length:        {0} in.'.format(round(self.l, 3)))
            print('Width:         {0} in.'.format(round(self.w, 3)))
        try:
            print('Thickness:     {0} in.'.format(round(self.t, 3)))
        except:
            print('THICKNESS ERROR')


class spiral:
    """
    blarg
    """
    def __init__(self,
                 pipe_ID,
                 blockage_ratio=44,
                 max_pressure_difference_atm=1,
                 pitch=None):  # ,
        #  GET RID OF               add_struts=False,
        #  GET RID OF               number_of_struts=None,
        #  GET RID OF               FS_struts=2,
        #  GET RID OF               strut_yield_psi=30000
        #                 ):

        # define the good stuff
        self.blockage_ratio = blockage_ratio
        self.pipe_ID = pipe_ID
        self.max_pressure_difference_atm = max_pressure_difference_atm

# GET RID OF THIS -------------------------------------------------------------
#        # strut your stuff!
#        strut = {}
#        strut['number'] = number_of_struts
#        strut['FS'] = FS_struts
#        strut['yield_psi'] = strut_yield_psi
#        self.strut = strut
#        if number_of_struts is not None and not add_struts:
#            print('You put a number of struts but didn''t change add_struts.')
#            print('Maybe try again with add_struts=True')
# -----------------------------------------------------------------------------

        # get diameter of spiral that results in the requested blockage ratio
        self.get_spiral_diameter()

# GET RID OF THIS -------------------------------------------------------------
#        # add struts to keep the sprial from bunching
#        if add_struts:
#            self.add_struts()
# -----------------------------------------------------------------------------

    def get_spiral_diameter(self):
        """
        waka waka
        """
        # calculate the ideal diameter
# GET RID OF THIS -------------------------------------------------------------
#        try:
#            # re-calculate spiral diameter if strut diameter is known
#            self.spiral_diameter = self.pipe_ID / 2 * \
#                (1 - np.sqrt(1 - self.blockage_ratio/100 +
#                             self.strut['number'] * (
#                                     self.strut['diameter'] / self.pipe_ID
#                                                     )**2))
#        except:
# -----------------------------------------------------------------------------
        #    # re-calculate spiral diameter without struts
        self.spiral_diameter = self.pipe_ID / 2 * \
            (1 - np.sqrt(1 - self.blockage_ratio/100))

        # get nearest fractional value
        nearest_fraction = 16
        self.spiral_diameter = Fraction(round(
                self.spiral_diameter * nearest_fraction)/nearest_fraction)

        # update blockage ratio
        self.get_blockage_ratio()

    def get_blockage_ratio(self, skip=True):
        """
        Re-calculates blockage ratio based on spiral diameter and number of
        struts
        """
        # calculate blockage ratio based on spiral
        self.blockage_ratio = (1 - (1 - 2 * float(self.spiral_diameter) /
                                    self.pipe_ID)**2) * 100
        try:
            # add blockage ratio due to struts
            self.blockage_ratio += 100*self.strut['number'] * (
                                                    self.strut['diameter'] /
                                                    self.pipe_ID
                                                               )**2
        except:
            pass

        self.get_run_up()

    def add_struts(self):
        """
        Adds a specified (and hopefully integer) number of struts in order to
        keep the shchelkin spiral from bunching up.
        """
        try:
            # if you can't make an integer out of the number_of_struts, you
            # really shouldn't be here.
            int(self.strut['number'])

            # make sure the user isn't too dumb to enter an integer number
            # of struts
            if not isinstance(self.strut['number'], int):
                # yell at the user for not using an integer
                print(
                      '{0} is not an integer number of struts, you dingus.'
                      .format(self.strut['number'])
                      )

                # make their input an integer
                self.strut['number'] = int(self.strut['number'])
                print(
                      'I''m just going to assume you meant {0}.'
                      .format(self.strut['number'])
                      )

            # calculate maximum force required to hold the spiral
            frontal_area = np.pi / 4 * \
                (
                 self.pipe_ID**2 - (
                                    self.pipe_ID -
                                    2 * float(self.spiral_diameter)
                                    )**2
                 )
            required_force = frontal_area * self.max_pressure_difference_atm \
                * 14.7

            # divide force among struts
            required_force = required_force / float(self.strut['number'])

            # calculate minimum strut diameter
            self.strut['diameter'] = np.sqrt(
                        4 * self.strut['FS'] * required_force /
                        (np.pi * self.strut['yield_psi'])
                        )

            # convert to next largest fraction
            nearest_fraction = 16
            self.strut['diameter'] = Fraction(
                                              np.ceil(
                                                      self.strut['diameter'] *
                                                      nearest_fraction
                                                      ) /
                                              nearest_fraction)

            # update spiral diameter and BR
            self.get_spiral_diameter()

        except:
            # if you can't make an integer out of the number_of_struts, Ice
            # Cube whoever's running this circus
            print('Chiggity check your number_of_struts. It''s jacked up.')

    def get_run_up(self, scale=1.1):
        """
        Approximates length of pipe required to cause a deflagration to
        develop into a detonation. Based on information from:

        Ciccarelli and Dorofeev, "Flame acceleration and transition to
        detonation in ducts," Progress in Energy and Combustion Science,
        vol. 34, issue 4, p537, 2008.

        NOTE:   The analysis in this function is LOOSELY based on Fig. 45 in
                the above article, using the curve for CH4BR since it shows the
                highest runup distance. THIS IS NOT CORRECT, as it is based on
                the use of a model which involves the sonic speed, laminar
                flame speed, and burned/unburned density ratio of the mixture
                in question. This is a FIRST PASS ESTIMATE ONLY, as it requires
                no information about the working fluid in question.

        INPUTS:
            blockage_ratio:     tube blockage ratio (%)
            scale:              (optional) amount to scale CH4BR curve, must be
                                greater than or equal to 1
                                default is 1.1 (110%)
        """
        # define information from fig. 45 in ciccarelli, CH4BR curve
        BR = np.array([0.01, 0.1, 0.2982, 0.7499]) * 100
        X_D = np.array([76.9388, 48.7755, 17.9592, 4.4898])

        runup = np.interp(self.blockage_ratio, BR, X_D) * scale * self.pipe_ID
        self.runup_length = runup

    def info(self, plot=False):
        """
        Returns information about the spiral
        """
        print()
        print('********************')
        print('* SHCHELKIN SPIRAL *')
        print('********************')
        print('Pipe ID:         {0} in.'.format(round(self.pipe_ID, 3)))
        print('Spiral Diameter: {0} in.'.format(self.spiral_diameter))
        print('Blockage Ratio:  {0} %'.format(round(self.blockage_ratio, 1)))
        print('Runup Length:    {0} in.'.format(round(self.runup_length, 2)))
        try:
            self.strut['diameter']
            print()
            print('***************************')
            print('* SHCHELKIN SPIRAL STRUTS *')
            print('***************************')
            print('Number:        {0}'.format(self.strut['number']))
            print('Diameter:      {0} in.'.format(self.strut['diameter']))
            print('Safety Factor: {0}'.format(self.strut['FS']))
        except:
            pass

        if plot:
            print('- NO PLOT FOR THIS COMPONENT -')


# DONE
#class flange:
#    """
#    asdga
#    """
#    def __init__(self,
#                 flange_name,
#                 max_pressure_atm,
#                 design_temperature_F=100):
#
#        # define given information
#        self.name = flange_name
#        self.max_pressure_atm = max_pressure_atm
#        self.design_temperature_F = design_temperature_F
#
#        # calculate required flange class
#        self.recalculate()
#
#    def recalculate(self):
#        # define possible flange classes
#        self.T = np.array([
#                0, 100, 200, 300, 400, 500, 600, 650, 700, 750, 800, 850, 900,
#                950, 1000
#                ])
#        flange_class = {}
#        flange_class['400'] = np.array([
#                               1000, 1000, 1000, 970, 940, 885, 805,
#                               785, 740, 675, 550, 425, 295, 185, 115
#                               ])
#        flange_class['600'] = np.array([
#                               1500, 1500, 1500, 1455, 1405, 1330,
#                               1210, 1175, 1110, 1015, 825, 640, 445,
#                               275, 170
#                               ])
#        flange_class['900'] = np.array([
#                               2250, 2250, 2250, 2185, 2110, 1995,
#                               1815, 1765, 1665, 1520, 1235, 955, 670,
#                               410, 255
#                               ])
#        flange_class['1500'] = np.array([
#                               3750, 3750, 3750, 3640, 3520, 3325,
#                               3025, 2940, 2775, 2535, 2055, 1595,
#                               1115, 685, 430
#                               ])
#        flange_class['2500'] = np.array([
#                               6250, 6250, 6250, 6070, 5865, 5540,
#                               5040, 4905, 4630, 4230, 3430, 2655,
#                               1855, 1145, 715
#                               ])
#
#        try:
#            # determine correct class such that the allowable pressure is
#            # higher than the required pressure at the given temperature
#            if self.design_temperature_F < 0:
#                print('Frosty.')
#            thePress = [np.interp(
#                                  self.design_temperature_F,
#                                  self.T, flange_class[key]
#                                  ) for key in flange_class]
#            theKeys = list(flange_class.keys())
#            indices = [i for i in range(len(thePress))
#                       if thePress[i] >= self.max_pressure_atm * 14.7]
#            index = str(min([int(theKeys[i]) for i in indices]))
#            self.flange_class = index
#            self.P = flange_class[index]
#        except:
#            print('Flange requirements outside of allowable range. Nerf it.')
#
#    def info(self, plot=False):
#        nameStr = 'Class '+self.flange_class+' Flange ('+self.name.title()+')'
#        print()
#        print('*' * (len(nameStr) + 4))
#        print('* {} *'.format(nameStr.upper()))
#        print('*' * (len(nameStr) + 4))
#
#        if plot:
#            plt.figure(nameStr)
#            plt.clf()
#            plt.plot(self.T, self.P / 14.7)
#            plt.grid('on')
#            plt.xlim([min(self.T), max(self.T)])
#            plt.xlabel('Flange Temperature (°F)')
#            plt.ylabel('Flange Max Pressure (atm)')
#            plt.title(nameStr)


class reflection:
    """
    write stuff here
    """
    def __init__(self,
                 P_max_atm=1,
                 P_out_max_atm=None,
                 P_relax_atm=None,
                 Vcj_m_s=2500,
                 a0_m_s=250,
                 gamma=1.14):
        # set variables
        self.Pr_atm = P_max_atm
        self.Vcj_m_s = Vcj_m_s
        self.a0_m_s = a0_m_s
        self.gamma = gamma
        self.P_out_max_atm = P_out_max_atm
        self.P_relax_atm = P_relax_atm

        # analyze!!
        self.analyze_reflection()

    def analyze_reflection(self):
        """
        do analysis wheeee
        """
        # calculate C-J pressure
        self.Pcj_atm = 4 * self.gamma * self.Pr_atm /\
            (5 * self.gamma + 1 + np.sqrt(17 * self.gamma**2 + 2 * self.gamma +
                                          1))

        # if no relaxation pressure was input, estimate as 0.4*Pcj
        if self.P_relax_atm is None:
            self.P_relax_atm = 0.4 * self.Pcj_atm

        # calculate initial pressure (atm)
        self.P0_atm = self.Pcj_atm * self.gamma /\
            (self.gamma + 1) * (self.a0_m_s / self.Vcj_m_s)

        # if no outlet pressure was input, estimate at 0.4*Pcj
        if self.P_out_max_atm is None:
            self.P_out_max_atm = (1.+1e-6)*self.P_relax_atm

        # calculate time constant tau based on a plot from Karnesky et al. 2013
        atm_to_bar = 1.01325
        self.tau = 24.95 * (self.P0_atm * atm_to_bar)**(-0.08222) + 285.9

        # convert tau to seconds
        self.tau /= 1e6

        # calculate minimum length
        self.L = -self.Vcj_m_s * self.tau * np.log(
                (self.P_out_max_atm - self.P_relax_atm) /
                (self.Pr_atm - self.P_relax_atm)
                )

    def info(self, plot=False):
        print()
        print('*************************')
        print('* DETONATION REFLECTION *')
        print('*************************')
        print('Max Allowable Reflection Pressure: {} atm'
              .format(round(max(self.Pr_atm), 2)))
        print('Max Allowable C-J Pressure:        {} atm'
              .format(round(max(self.Pcj_atm), 2)))
        print('Minimum Length:                    {} in.'
              .format(round(max(self.L), 2),))
        if len(self.L) > 1 and plot:
            nameStr = 'Reflection Decay Length vs. Initial Pressure'
            plt.figure(nameStr)
            plt.plot(self.P0_atm, self.L)
            plt.grid('on')
            plt.xlim([min(self.P0_atm), max(self.P0_atm)])
            plt.xlabel('Initial Pressure (atm)')
            plt.ylabel('Reflection Decay Length (in)')
            nameStr = nameStr + \
                '\n$\gamma$ = {0}, $V_{{C-J}}$ = {1} m/s , ' \
                .format(self.gamma, self.Vcj_m_s) + \
                '$a_{{0}}$ = {0} m/s' \
                .format(self.a0_m_s)
            plt.title(nameStr)


class boltPattern:
    """
    Calculates stresses for bolt patterns
    """
    def __init__(self,
                 N_bolts=np.array([20]),
                 bolt_size='1/4-20',
                 bolt_class='2A',
                 hole_class='2B',
                 plate_max_stress_psi=30000,
                 plate_temp_derating='stainless',
                 bolt_max_stress_psi=120000,
                 bolt_temp_derating='stainless',
                 engagement_length_inches=0.5,
                 temp_F=np.array([70]),
                 total_force_lbf=np.array([70367]),
                 desired_FS=3):

        # Read in bolt dimensions from csv
        self.bolt_info = pd.read_csv('bolts.csv', index_col=0)
        self.internal = pd.read_csv('int_thread.csv', index_col=0)
        self.external = pd.read_csv('ext_thread.csv', index_col=0)

        # initialize results dictionaries
        self.bolt = dict()
        self.plate = dict()

        # Make sure that temperature and pressure are the same size
        if len(temp_F) != len(total_force_lbf):
            raise Exception('Your P and T inputs are different lengths.')
        else:
            # initialize blank arrays
            self.bolt['A'] = np.ones(len(temp_F))
            self.bolt['FS'] = np.ones(len(temp_F))
            self.plate['A'] = np.ones(len(temp_F))
            self.plate['FS'] = np.ones(len(temp_F))

            for i in range(len(temp_F)):
                # calculate stuff at each T,P,N_bolts combo
                values = self.areaCalc(N_bolts[i],
                                       bolt_size,
                                       bolt_class,
                                       hole_class,
                                       bolt_max_stress_psi,
                                       engagement_length_inches)
                self.bolt['A'][i] = values[0]
                self.plate['A'][i] = values[1]

                values = self.stressCalc(temp_F[i],
                                         bolt_temp_derating,
                                         plate_temp_derating,
                                         bolt_max_stress_psi,
                                         plate_max_stress_psi,
                                         N_bolts[i],
                                         total_force_lbf[i],
                                         self.bolt['A'][i],
                                         self.plate['A'][i])
                self.bolt['FS'][i] = values[0]
                self.plate['FS'][i] = values[1]

    def areaCalc(self,
                 N_bolts,
                 bolt_size,
                 bolt_class,
                 hole_class,
                 bolt_max_stress_psi,
                 L_e):

        # get bolt information
        n = float(self.bolt_info['TPI'][bolt_size])

        # calculate bolt stress area
        # http://www.engineersedge.com/thread_stress_area_a.htm < 100 ksi
        # http://www.engineersedge.com/thread_stress_area_b.htm > 100 ksi
        if bolt_max_stress_psi < 100000:
            D = self.threadInfo(self.external,
                                bolt_size,
                                bolt_class,
                                'd_basic')
            bolt_A = np.pi / 4 * (D - 0.9743 / n)**2
        else:
            E_s_min = self.threadInfo(self.external,
                                      bolt_size,
                                      bolt_class,
                                      'd_pitch_min')
            bolt_A = np.pi * (E_s_min / 2 - 0.16238 / n)**2

        # calculate plate stress area
        # http://www.engineersedge.com/thread_strength/thread_bolt_stress.htm
        E_n_max = self.threadInfo(self.internal,
                                  bolt_size,
                                  hole_class,
                                  'd_pitch_max')
        D_s_min = self.threadInfo(self.internal,
                                  bolt_size,
                                  hole_class,
                                  'd_major_min')
        plate_A = np.pi * n * L_e * D_s_min * (1 / (2 * n) +
                                               0.57735 * (D_s_min - E_n_max))
        return([bolt_A, plate_A])

    def stressCalc(self,
                   temp_F,
                   bolt_temp_derating,
                   plate_temp_derating,
                   s_bolt,
                   s_plate,
                   N_bolts,
                   F,
                   A_bolt,
                   A_plate):

        # get thermal adjustment factors
        k_bolt = self.thermalKnockdown(temp_F, bolt_temp_derating)
        k_plate = self.thermalKnockdown(temp_F, plate_temp_derating)

        # calculate allowable stresses
        s_allow_bolt = k_bolt * s_bolt
        s_allow_plate = k_plate * s_plate

        # adjust total force to per-bolt basis
        F = F / N_bolts

        # find actual stress
        s_actual_bolt = F / A_bolt
        s_actual_plate = F / A_plate

        # calculate safety factors
        return([s_allow_bolt / s_actual_bolt,
                s_allow_plate / s_actual_plate])

    def threadInfo(self,
                   inputFrame,
                   threadSize,
                   thread_class,
                   desiredInfo):
        theOutput = pd.DataFrame(inputFrame[desiredInfo][threadSize].values,
                                 inputFrame['thread_class'][threadSize].values,
                                 columns=[desiredInfo])
        theOutput = theOutput[desiredInfo][thread_class]
        return(theOutput)

    def thermalKnockdown(self,
                         temp_F,
                         material='stainless'):
        # massage string around
        material = 'material_' + material + '.csv'

        # import thermal data from csv and strip off head row
        # 0 is max stress scaling factor
        # 1 is temperature
        scaleData = np.genfromtxt(material, delimiter=',')[1:, :]

        return(np.interp(temp_F, scaleData[:, 1], scaleData[:, 0]))


if __name__ == '__main__':
    print('''DANGER! DANGER, WILL ROBINSON!!!
This is not the analysis file!
This is only the toolbox!
ARRRRRRRRRGH!!''')
