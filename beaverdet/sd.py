# -*- coding: utf-8 -*-
"""
Functions for detonation calculations.

Original functions from Shock and Detonation Toolbox
http://www.galcit.caltech.edu/EDL/public/cantera/html/SD_Toolbox/
"""

import warnings
import numpy as np
import cantera as ct
import multiprocessing as mp
from scipy.optimize import curve_fit


class Detonation:
    @classmethod
    def cj_state(
            cls,
            working_gas,
            initial_state_gas,
            error_tol_temperature,
            error_tol_velocity,
            density_ratio,
            max_iterations=500
    ):
        """
        This function calculates the Chapman-Jouguet state and wave speed using
        Reynolds' iterative method.

        Original function: CJ_calc in PostShock.py

        Parameters
        ----------
        working_gas : cantera.composite.Solution
            A cantera gas object used for calculations (???).
        initial_state_gas : cantera.composite.Solution
            A cantera gas object for the working gas mixture in its initial,
            undetonated state.
        error_tol_temperature : float
            Temperature error tolerance for iteration.
        error_tol_velocity : float
            Velocity error tolerance for iteration.
        density_ratio : float
            density ratio.
        max_iterations : int
            Maximum number of loop iterations used to calculate output. Default
            is 500.

        Returns
        -------
        working_gas : cantera.composite.Solution
            Gas object at equilibrium state.
        initial_velocity : float
            Initial velocity resulting in the input density ratio, in m/s.
        """
        # initial state
        initial_volume = 1 / initial_state_gas.density

        # set guess values
        guess_temperature = 2000
        guess_velocity = 2000
        guess_volume = initial_volume / density_ratio
        guess_density = 1 / guess_volume

        # set deltas
        delta_temperature = 1000
        delta_velocity = 1000

        # equilibrate
        Properties.equilibrium(
            working_gas,
            guess_density,
            guess_temperature
        )

        loop_counter = 0
        while (
                abs(delta_temperature) > (error_tol_temperature *
                                          guess_temperature)
                or
                abs(delta_velocity) > (error_tol_velocity *
                                       guess_velocity)
        ):
            loop_counter += 1
            # check for non-convergence
            if loop_counter == max_iterations:
                warnings.warn(
                    'No convergence within {0} iterations'.format(
                        max_iterations
                    ),
                    Warning
                )
                return [working_gas, guess_velocity]

            # calculate unperturbed enthalpy and press. error for current guess
            [error_enthalpy,
             error_pressure] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                guess_velocity
            )

            # perturb temperature
            delta_temperature = 0.02 * guess_temperature
            perturbed_temperature = guess_temperature + delta_temperature
            Properties.equilibrium(
                working_gas,
                guess_density,
                perturbed_temperature
            )

            # calculate error rates for temperature perturbed state
            [error_perturbed_enthalpy,
             error_perturbed_pressure] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                guess_velocity
            )
            derivative_enthalpy_temperature = (
                error_perturbed_enthalpy - error_enthalpy
                                              ) / delta_temperature
            derivative_pressure_temperature = (
                error_perturbed_pressure - error_pressure
                                              ) / delta_temperature

            # perturb velocity
            delta_velocity = 0.02 * guess_velocity
            perturbed_velocity = guess_velocity + delta_velocity
            perturbed_temperature = guess_temperature
            Properties.equilibrium(
                working_gas,
                guess_density,
                perturbed_temperature
            )

            # calculate error rates for velocity perturbed state
            [error_perturbed_enthalpy,
             error_perturbed_pressure] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                perturbed_velocity
            )
            derivative_enthalpy_velocity = (error_perturbed_enthalpy -
                                            error_enthalpy) / delta_velocity
            derivative_pressure_velocity = (error_perturbed_pressure -
                                            error_pressure) / delta_velocity

            # invert matrix
            jacobian = derivative_enthalpy_temperature *\
                derivative_pressure_velocity - \
                derivative_pressure_temperature * \
                derivative_enthalpy_velocity
            b = [derivative_pressure_velocity,
                 -derivative_enthalpy_velocity,
                 -derivative_pressure_temperature,
                 derivative_enthalpy_temperature]
            a = [-error_enthalpy,
                 -error_pressure]
            delta_temperature = (b[0] * a[0] + b[1] * a[1]) / jacobian
            delta_velocity = (b[2] * a[0] + b[3] * a[1]) / jacobian

            # limit temperature changes
            max_temperature_delta = 0.2 * guess_temperature
            if abs(delta_temperature) > max_temperature_delta:
                delta_temperature *= max_temperature_delta / \
                                     abs(delta_temperature)

            # apply deltas and equilibrate
            guess_temperature += delta_temperature
            guess_velocity += delta_velocity
            Properties.equilibrium(
                working_gas,
                guess_density,
                guess_temperature
            )

        return [working_gas, guess_velocity]

    @classmethod
    def _calculate_over_ratio_range(
            cls,
            current_state_number,
            current_density_ratio,
            initial_temperature,
            initial_pressure,
            species_mole_fractions,
            mechanism,
            error_tol_temperature,
            error_tol_velocity
    ):
        initial_state_gas = ct.Solution(mechanism)
        initial_state_gas.TPX = [
            initial_temperature,
            initial_pressure,
            species_mole_fractions
        ]
        working_gas = ct.Solution(mechanism)
        working_gas.TPX = [
            initial_temperature,
            initial_pressure,
            species_mole_fractions
        ]
        [_,
         current_velocity] = cls.cj_state(
            working_gas,
            initial_state_gas,
            error_tol_temperature,
            error_tol_velocity,
            current_density_ratio
        )

        return current_state_number, current_velocity

    @classmethod
    def cj_speed(
            cls,
            initial_pressure,
            initial_temperature,
            species_mole_fractions,
            mechanism,
            use_multiprocessing=False,
            return_r_squared=False,
            return_state=False
    ):
        """
        This function calculates CJ detonation velocity

        Original function: CJspeed in PostShock.py

        Parameters
        ----------
        initial_pressure : float
            initial pressure (Pa)
        initial_temperature : float
            initial temperature (K)
        species_mole_fractions : str or dict
            string or dictionary of reactant species mole fractions
        mechanism : str
            cti file containing mechanism data (e.g. 'gri30.cti')
        use_multiprocessing : bool
            use multiprocessing to speed up CJ speed calculation
        return_r_squared : bool
            return the R^2 value of the CJ speed vs. density ratio fit
        return_state : bool
            return the CJ state corresponding to the calculated velocity

        Returns
        -------
        dict
        """
        # DECLARATIONS
        num_steps = 20
        max_density_ratio = 2.0
        min_density_ratio = 1.5

        if use_multiprocessing:
            pool = mp.Pool()

        # Set error tolerances for CJ state calculation
        error_tol_temperature = 1e-4
        error_tol_velocity = 1e-4

        counter = 1
        r_squared = 0.0
        delta_r_squared = 0.0
        adjusted_density_ratio = 0.0

        def curve_fit_function(x, a, b, c):
            """
            Quadratic function for least-squares curve fit of cj speed vs.
            density ratio
            """
            return a * x**2 + b * x + c

        while (counter <= 4) and (
                (r_squared < 0.99999) or (delta_r_squared < 1e-7)
        ):
            density_ratio_array = np.linspace(
                min_density_ratio,
                max_density_ratio,
                num_steps + 1
            )

            if use_multiprocessing:
                # parallel loop through density ratios
                stargs = [[number,
                           ratio,
                           initial_temperature,
                           initial_pressure,
                           species_mole_fractions,
                           mechanism,
                           error_tol_temperature,
                           error_tol_velocity
                           ]
                          for number, ratio in
                          zip(
                              range(len(density_ratio_array)),
                              density_ratio_array
                          )
                          ]
                # noinspection PyUnboundLocalVariable
                result = pool.starmap(cls._calculate_over_ratio_range, stargs)

            else:
                # no multiprocessing, just use map
                result = list(map(
                    cls._calculate_over_ratio_range,
                    [item for item in range(len(density_ratio_array))],
                    density_ratio_array,
                    [initial_temperature for _ in density_ratio_array],
                    [initial_pressure for _ in density_ratio_array],
                    [species_mole_fractions for _ in density_ratio_array],
                    [mechanism for _ in density_ratio_array],
                    [error_tol_temperature for _ in density_ratio_array],
                    [error_tol_velocity for _ in density_ratio_array]
                ))

            result.sort()
            cj_velocity_calculations = np.array(
                [item for (_, item) in result]
            )

            # Get curve fit
            [curve_fit_coefficients, _] = curve_fit(
                curve_fit_function,
                density_ratio_array,
                cj_velocity_calculations
                )

            # Calculate R^2 value
            residuals = cj_velocity_calculations - curve_fit_function(
                density_ratio_array,
                *curve_fit_coefficients
                )
            old_r_squared = r_squared
            r_squared = 1 - (
                np.sum(residuals**2) /
                np.sum(
                    (
                        cj_velocity_calculations -
                        np.mean(cj_velocity_calculations)
                        )**2
                    )
                )
            delta_r_squared = abs(old_r_squared - r_squared)

            adjusted_density_ratio = (
                -curve_fit_coefficients[1] /
                (2. * curve_fit_coefficients[0])
                )
            min_density_ratio = adjusted_density_ratio * (1 - 0.001)
            max_density_ratio = adjusted_density_ratio * (1 + 0.001)
            counter += 1

        # noinspection PyUnboundLocalVariable
        cj_speed = curve_fit_function(
            adjusted_density_ratio,
            *curve_fit_coefficients
            )

        if return_state:
            initial_state_gas = ct.Solution(mechanism)
            working_gas = ct.Solution(mechanism)
            initial_state_gas.TPX = [
                initial_temperature,
                initial_pressure,
                species_mole_fractions
            ]
            working_gas.TPX = [
                initial_temperature,
                initial_pressure,
                species_mole_fractions
            ]
            cls.cj_state(
                working_gas,
                initial_state_gas,
                error_tol_temperature,
                error_tol_velocity,
                adjusted_density_ratio
            )

        if return_r_squared and return_state:
            # noinspection PyUnboundLocalVariable
            return {'cj speed': cj_speed,
                    'R^2': r_squared,
                    'cj state': working_gas
                    }
        elif return_state:
            # noinspection PyUnboundLocalVariable
            return {'cj speed': cj_speed,
                    'cj state': working_gas
                    }
        elif return_r_squared:
            return {'cj speed': cj_speed,
                    'R^2': r_squared
                    }
        else:
            return {'cj speed': cj_speed}


class Properties:
    @staticmethod
    def equilibrium(
            gas,
            density,
            temperature
    ):
        """
        This function calculates the equilibrium pressure and enthalpy given
        temperature and density

        Original function: eq_state in Thermo.py

        Parameters
        ----------
        gas : cantera.composite.Solution
            Working gas object.
        density : float
            Mixture density in kg/m^3.
        temperature : float
            Mixture temperature in K.

        Returns
        -------
        dict
            Dictionary containing pressure and temperature values
        """

        gas.TD = temperature, density
        gas.equilibrate('TV')
        pressure = gas.P
        enthalpy = gas.enthalpy_mass

        return {'pressure': pressure, 'enthalpy': enthalpy}

    @staticmethod
    def frozen(
            gas,
            density,
            temperature
    ):
        """
        This function calculates the frozen pressure and enthalpy given
        temperature and density

        Original function: state in Thermo.py

        Parameters
        ----------
        gas : cantera.composite.Solution
            Working gas object.
        density : float
            Mixture density in kg/m^3.
        temperature : float
            Mixture temperature in K.

        Returns
        -------
        dict
            Dictionary containing pressure and temperature values
        """

        gas.TD = temperature, density
        pressure = gas.P
        enthalpy = gas.enthalpy_mass

        return {'pressure': pressure, 'enthalpy': enthalpy}


class GetError:
    @staticmethod
    def equilibrium(
            working_gas,
            initial_state_gas,
            initial_velocity_guess
    ):
        """
        This function uses the momentum and energy conservation equations to
        calculate error in current pressure and enthalpy guesses. In this case,
        working state is in equilibrium.

        Original function: FHFP_CJ in PostShock.py

        Parameters
        ----------
        working_gas : cantera.composite.Solution
            A cantera gas object used for calculations (???).
        initial_state_gas : cantera.composite.Solution
            A cantera gas object for the working gas mixture in its initial,
            undetonated state.
        initial_velocity_guess : float
            A guess for the initial velocity in m/s

        Returns
        -------
        list
            A list of errors in [enthalpy, pressure]
        """

        initial = {
            'pressure': initial_state_gas.P,
            'enthalpy': initial_state_gas.enthalpy_mass,
            'density': initial_state_gas.density,
            'velocity': initial_velocity_guess
        }

        working = {
            'pressure': working_gas.P,
            'enthalpy': working_gas.enthalpy_mass,
            'density': working_gas.density
        }

        working['velocity'] = initial['velocity'] * (
                initial['density'] / working['density']
        )

        squared_velocity = {
            'initial': initial['velocity']**2,
            'working': working['velocity']**2
        }

        enthalpy_error = (
                (working['enthalpy'] + 0.5 * squared_velocity['working']) -
                (initial['enthalpy'] + 0.5 * squared_velocity['initial'])
        )

        pressure_error = (
                (
                        working['pressure'] +
                        working['density'] * squared_velocity['working']
                ) - (
                        initial['pressure'] +
                        initial['density'] * squared_velocity['initial']
                )
        )

        return [enthalpy_error, pressure_error]

    @staticmethod
    def frozen(
            working_gas,
            initial_state_gas,
            initial_velocity_guess
    ):
        """
        This function uses the momentum and energy conservation equations to
        calculate error in current pressure and enthalpy guesses. In this case,
        working state is frozen.

        Original function: FHFP_CJ in PostShock.py

        NOTE: this function is identical to equilibrium...

        Do you want to build a snowman?

        Parameters
        ----------
        working_gas : cantera.composite.Solution
            A cantera gas object used for calculations.
        initial_state_gas : cantera.composite.Solution
            A cantera gas object for the working gas mixture in its initial,
            undetonated state.
        initial_velocity_guess : float
            A guess for the initial velocity in m/s

        Returns
        -------
        list
            A list of errors in [enthalpy, pressure]
        """

        initial = {
            'pressure': initial_state_gas.P,
            'enthalpy': initial_state_gas.enthalpy_mass,
            'density': initial_state_gas.density,
            'velocity': initial_velocity_guess
        }

        working = {
            'pressure': working_gas.P,
            'enthalpy': working_gas.enthalpy_mass,
            'density': working_gas.density
        }

        working['velocity'] = initial['velocity'] * (
                initial['density'] / working['density']
        )

        squared_velocity = {
            'initial': initial['velocity']**2,
            'working': working['velocity']**2
        }

        enthalpy_error = (
                (working['enthalpy'] + 0.5 * squared_velocity['working']) -
                (initial['enthalpy'] + 0.5 * squared_velocity['initial'])
        )

        pressure_error = (
                (
                        working['pressure'] +
                        working['density'] * squared_velocity['working']
                ) - (
                        initial['pressure'] +
                        initial['density'] * squared_velocity['initial']
                )
        )

        return [enthalpy_error, pressure_error]

    @staticmethod
    def reflected_shock_frozen(
            shock_speed,
            working_gas,
            post_shock_gas
    ):
        """
        This function uses the momentum and energy conservation equations to
        calculate error in current pressure and enthalpy guesses during
        reflected shock calculations. In this case, working state is frozen.

        Original function: FHFP_reflected_fr in reflections.py

        Parameters
        ----------
        shock_speed : float
            Current post-incident-shock lab frame particle speed
        working_gas : cantera.composite.Solution
            A cantera gas object used for calculations.
        post_shock_gas : cantera.composite.Solution
            A cantera gas object at post-incident-shock state (already computed)

        Returns
        -------
        numpy array
            A numpy array of errors in [enthalpy, pressure]
        """
        post_shock = {
            'pressure': post_shock_gas.P,
            'enthalpy': post_shock_gas.enthalpy_mass,
            'density': post_shock_gas.density
            }

        working = {
            'pressure': working_gas.P,
            'enthalpy': working_gas.enthalpy_mass,
            'density': working_gas.density
            }

        enthalpy_error = (
            working['enthalpy'] -
            post_shock['enthalpy'] -
            0.5 * (shock_speed**2)*(
                    (working['density'] / post_shock['density']) + 1
                ) /
            (working['density'] / post_shock['density'] - 1)
            )

        pressure_error = (
            working['pressure'] -
            post_shock['pressure'] -
            working['density'] *
            (shock_speed**2) / (
                working['density'] /
                post_shock['density']-1
                )
            )

        return [enthalpy_error, pressure_error]


class Reflection:
    @classmethod
    def reflect(
            cls,
            initial_state_gas,
            post_shock_gas,
            working_gas,
            incident_shock_speed
    ):
        """
        This function calculates equilibrium post-reflected-shock state assuming
        u1 = 0

        reflected_eq

        Parameters
        ----------
        initial_state_gas : cantera.composite.Solution
            gas object at initial state
        post_shock_gas : cantera.composite.Solution
            gas object at post-incident-shock state (already computed)
        working_gas : cantera.composite.Solution
            working gas object
        incident_shock_speed : float
            incident shock speed (m/s)

        Returns
        -------
        working['pressure'] : float
            post-reflected-shock pressure (Pa)
        reflected_shock_speed : float
            reflected shock speed (m/s)
        working_gas : cantera.composite.Solution
            gas object at equilibrium post-reflected-shock state
        """
        initial = {
            'pressure': initial_state_gas.P,
            'volume': 1 / initial_state_gas.density
        }

        reflected = {
            'pressure': post_shock_gas.P,
            'density': post_shock_gas.density,
            'volume': 1 / post_shock_gas.density,
            'temperature': post_shock_gas.T
        }
        reflected['velocity'] = np.sqrt(
            (reflected['pressure'] - initial['pressure']) *
            (initial['volume'] - reflected['volume'])
        )

        working = {
            'volume': 0.2 / reflected['density']
        }
        working['pressure'] = (
                reflected['pressure'] +
                reflected['density'] *
                (incident_shock_speed ** 2) *
                (1 - working['volume'] / reflected['volume'])
        )
        working['temperature'] = (
                reflected['temperature'] *
                working['pressure'] *
                working['volume'] /
                (reflected['pressure'] * reflected['volume'])
        )

        working_gas.TPX = [
            working['temperature'],
            working['pressure'],
            post_shock_gas.X
        ]
        working_gas = cls.get_reflected_eq_state(
            reflected['velocity'],
            post_shock_gas,
            working_gas
        )
        working['pressure'] = working_gas.P
        reflected_shock_speed = (
                (working['pressure'] - reflected['pressure']) /
                reflected['velocity'] /
                reflected['density'] -
                reflected['velocity']
        )

        return [working['pressure'], reflected_shock_speed, working_gas]

    @staticmethod
    def get_reflected_eq_state(
            particle_speed,
            post_shock_gas,
            working_gas,
            error_tol_temperature=1e-4,
            error_tol_specific_volume=1e-4,
            max_iterations=500
    ):
        """
        This function calculates equilibrium post-reflected-shock state for a
        specified shock velocity

        Original function: PostReflectedShock_eq in reflections.py

        Parameters
        ----------
        particle_speed : float
            current post-incident-shock lab frame particle speed
        post_shock_gas : cantera.composite.Solution
            gas object at post-incident-shock state (already computed)
        working_gas : cantera.composite.Solution
            working gas object
        error_tol_temperature : float
            Temperature error tolerance for iteration.
        error_tol_specific_volume : float
            Specific volume error tolerance for iteration.
        max_iterations : int
            maximum number of loop iterations

        Returns
        -------
        working_gas : cantera.composite.Solution
            gas object at equilibrium post-reflected-shock state
        """

        # set post-shocked state
        post_shock = dict()
        post_shock['volume'] = 1 / post_shock_gas.density

        # set reflected guess state
        guess = dict()
        guess['temperature'] = working_gas.T
        guess['density'] = working_gas.density
        guess['volume'] = 1 / guess['density']

        # set initial delta guesses
        delta = dict()
        delta['temperature'] = 1000
        delta['volume'] = 1000

        # equilibrate at guess state
        Properties.equilibrium(
            working_gas,
            guess['density'],
            guess['temperature']
        )

        # calculate reflected state
        loop_counter = 0
        while (
                (
                        abs(delta['temperature'])
                        >
                        error_tol_temperature * guess['temperature'])
                or
                (
                        abs(delta['volume'])
                        >
                        error_tol_specific_volume * guess['volume']
                )
        ):
            loop_counter += 1
            if loop_counter == max_iterations:
                warnings.warn(
                    'Calculation did not converge for U = {0:.2f} ' +
                    'after {1} iterations'.format(particle_speed, loop_counter))
                return working_gas

            # calculate enthalpy and pressure error for current guess
            [err_enthalpy,
             err_pressure] = GetError.reflected_shock_frozen(
                particle_speed,
                working_gas,
                post_shock_gas
            )

            # equilibrate working gas with perturbed temperature
            delta['temperature'] = guess['temperature'] * 0.02
            # equilibrate temperature perturbed state
            Properties.equilibrium(
                working_gas,
                guess['density'],
                guess['temperature'] + delta['temperature']
            )

            # calculate enthalpy and pressure error for perturbed temperature
            [err_enthalpy_perturbed,
             err_pressure_perturbed] = GetError.reflected_shock_frozen(
                particle_speed,
                working_gas,
                post_shock_gas
            )

            # calculate temperature derivatives
            deriv_enthalpy_temperature = (err_enthalpy_perturbed -
                                          err_enthalpy) / delta['temperature']
            deriv_pressure_temperature = (err_pressure_perturbed -
                                          err_pressure) / delta['temperature']

            # equilibrate working gas with perturbed volume
            delta['volume'] = 0.02 * guess['volume']
            # equilibrate volume perturbed state
            Properties.equilibrium(
                working_gas,
                1 / (guess['volume'] + delta['volume']),
                guess['temperature']
            )

            # calculate enthalpy and pressure error for perturbed specific vol
            [err_enthalpy_perturbed,
             err_pressure_perturbed] = GetError.reflected_shock_frozen(
                particle_speed,
                working_gas,
                post_shock_gas
            )

            # calculate specific volume derivatives
            deriv_enthalpy_volume = (err_enthalpy_perturbed -
                                     err_enthalpy) / delta['volume']
            deriv_pressure_volume = (err_pressure_perturbed -
                                     err_pressure) / delta['volume']

            # solve matrix for temperature and volume deltas
            jacobian = (
                    deriv_enthalpy_temperature *
                    deriv_pressure_volume -
                    deriv_pressure_temperature *
                    deriv_enthalpy_volume
            )
            bb = [
                deriv_pressure_volume,
                -deriv_enthalpy_volume,
                -deriv_pressure_temperature,
                deriv_enthalpy_temperature
            ]
            aa = [-err_enthalpy, -err_pressure]
            delta['temperature'] = (bb[0] * aa[0] +
                                    bb[1] * aa[1]) / jacobian
            delta['volume'] = (bb[2] * aa[0] +
                               bb[3] * aa[1]) / jacobian

            # check and limit temperature delta
            delta['temp_max'] = 0.2 * guess['temperature']
            if abs(delta['temperature']) > delta['temp_max']:
                delta['temperature'] = (
                        delta['temp_max'] *
                        delta['temperature'] /
                        abs(delta['temperature'])
                )

            # check and limit specific volume delta
            perturbed_volume = guess['volume'] + delta['volume']
            if perturbed_volume > post_shock['volume']:
                delta['volume_max'] = 0.5 * (
                            post_shock['volume'] - guess['volume'])
            else:
                delta['volume_max'] = 0.2 * guess['volume']

            if abs(delta['volume']) > delta['volume_max']:
                delta['volume'] = (
                        delta['volume_max'] *
                        delta['volume'] /
                        abs(delta['volume'])
                )

            # apply calculated and limited deltas to temperature and spec. vol
            guess['temperature'] += + delta['temperature']
            guess['volume'] += delta['volume']
            guess['density'] = 1 / guess['volume']

            # equilibrate working gas with updated state
            Properties.equilibrium(
                working_gas,
                guess['density'],
                guess['temperature']
            )

        return working_gas

    @staticmethod
    def get_post_shock_eq_state(
            wave_speed,
            initial_pressure,
            initial_temperature,
            reactant_mixture,
            mechanism,
            error_tol_temperature=1e-4,
            error_tol_specific_volume=1e-4,
            max_iterations=500
    ):
        """
        This function calculates equilibrium post-shock state using Reynolds'
        iterative method

        Original functions: shk_eq_calc and PostShock_eq in reflections.py

        Parameters
        ----------
        wave_speed : float
            speed at which the shock is traveling
        initial_pressure : float
            Pressure of initial state mixture (Pa)
        initial_temperature : float
            Temperature of initial state mixture (K)
        reactant_mixture : str or dict
            String or dict of reactant species moles or mole fractions
        mechanism : str
            Mechanism file to use (e.g. 'gri30.cti')
        error_tol_temperature : float
            Temperature error tolerance for iteration.
        error_tol_specific_volume : float
            Specific volume error tolerance for iteration.
        max_iterations : int
            maximum number of loop iterations

        Returns
        -------
        working_gas : cantera.composite.Solution
            gas object at equilibrium post-reflected-shock state
        """
        # set gas objects
        initial_state_gas = ct.Solution(mechanism)
        initial_state_gas.TPX = [
            initial_temperature,
            initial_pressure,
            reactant_mixture
        ]
        working_gas = ct.Solution(mechanism)
        working_gas.TPX = [
            initial_temperature,
            initial_pressure,
            reactant_mixture
        ]

        # set initial state variables
        initial = dict()
        initial['density'] = initial_state_gas.density
        initial['volume'] = 1 / initial['density']
        initial['pressure'] = initial_pressure
        initial['temperature'] = initial_temperature

        # set initial delta guess
        delta = dict()
        delta['temperature'] = 1000
        delta['volume'] = 1000

        # set guess state variables
        guess = dict()
        guess['volume'] = 0.2 * initial['volume']
        guess['density'] = 1 / guess['volume']
        guess['pressure'] = (
                initial['pressure'] +
                initial['density'] *
                (wave_speed ** 2) *
                (1 - guess['volume'] / initial['volume'])
        )
        guess['temperature'] = (
                initial['temperature'] *
                guess['pressure'] *
                guess['volume'] /
                (initial['pressure'] * initial['volume'])
        )

        # equilibrate working gas
        Properties.equilibrium(
            working_gas,
            guess['density'],
            guess['temperature']
        )

        # calculate equilibrium state
        loop_counter = 0
        while (
                (
                        abs(delta['temperature'])
                        >
                        error_tol_temperature * guess['temperature']
                )
                or
                (
                        abs(delta['volume'])
                        >
                        error_tol_specific_volume * guess['volume']
                )
        ):
            loop_counter += 1
            if loop_counter == max_iterations:
                warnings.warn(
                    "No convergence in {0} iterations".format(loop_counter)
                )
                return working_gas

            # calculate enthalpy and pressure error for current guess
            [err_enthalpy, err_pressure] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                wave_speed
            )

            # equilibrate working gas with perturbed temperature
            delta['temperature'] = 0.02 * guess['temperature']
            Properties.equilibrium(
                working_gas,
                guess['density'],
                guess['temperature'] + delta['temperature']
            )

            # calculate enthalpy and pressure error for perturbed temperature
            [err_enthalpy_perturbed,
             err_pressure_perturbed] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                wave_speed
            )

            # calculate temperature derivatives
            deriv_enthalpy_temperature = (err_enthalpy_perturbed -
                                          err_enthalpy) / delta['temperature']
            deriv_pressure_temperature = (err_pressure_perturbed -
                                          err_pressure) / delta['temperature']

            # equilibrate working gas with perturbed volume
            delta['volume'] = 0.02 * guess['volume']
            Properties.equilibrium(
                working_gas,
                1 / (guess['volume'] + delta['volume']),
                guess['temperature']
            )

            # calculate enthalpy and pressure error for perturbed specific vol
            [err_enthalpy_perturbed,
             err_pressure_perturbed] = GetError.equilibrium(
                working_gas,
                initial_state_gas,
                wave_speed
            )

            # calculate specific volume derivatives
            deriv_enthalpy_volume = (err_enthalpy_perturbed -
                                     err_enthalpy) / delta['volume']
            deriv_pressure_volume = (err_pressure_perturbed -
                                     err_pressure) / delta['volume']

            # solve matrix for temperature and volume deltas
            jacobian = (
                    deriv_enthalpy_temperature *
                    deriv_pressure_volume -
                    deriv_pressure_temperature *
                    deriv_enthalpy_volume
            )
            bb = [
                deriv_pressure_volume,
                -deriv_enthalpy_volume,
                -deriv_pressure_temperature,
                deriv_enthalpy_temperature
            ]
            aa = [-err_enthalpy, -err_pressure]
            delta['temperature'] = (bb[0] * aa[0] +
                                    bb[1] * aa[1]) / jacobian
            delta['volume'] = (bb[2] * aa[0] +
                               bb[3] * aa[1]) / jacobian

            # check and limit temperature delta
            delta['temp_max'] = 0.2 * guess['temperature']
            if abs(delta['temperature']) > delta['temp_max']:
                delta['temperature'] = delta['temp_max'] * \
                                       delta['temperature'] / \
                                       abs(delta['temperature'])

            # check and limit specific volume delta
            perturbed_volume = guess['volume'] + delta['volume']
            if perturbed_volume > initial['volume']:
                delta['volume_max'] = 0.5 * (
                            initial['volume'] - guess['volume'])
            else:
                delta['volume_max'] = 0.2 * guess['volume']
            if abs(delta['volume']) > delta['volume_max']:
                delta['volume'] = delta['volume_max'] * \
                                  delta['volume'] / \
                                  abs(delta['volume'])

            # apply calculated and limited deltas to temperature and spec. vol
            guess['temperature'] += delta['temperature']
            guess['volume'] += delta['volume']
            guess['density'] = 1 / guess['volume']

            # equilibrate working gas with updated state
            Properties.equilibrium(
                working_gas,
                guess['density'],
                guess['temperature']
            )

        return working_gas
