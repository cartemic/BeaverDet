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


# noinspection SpellCheckingInspection
def cj_curve_fit(x, y):
    """
        Determines least squares fit of parabolic data.
        Original function: LSQ_CJspeed from sdtoolbox, but vectorized

        Parameters
        ----------
        x
            iterable with independent data points for curve fitting
        y
            iterable with dependent data points for curve fitting

        Returns
        -------
        tuple
            a, b, c, r_squared where:
            a, b, c = coefficients of quadratic function (ax^2 + bx + c = 0)
        """
    # enforce numpy and float dtype
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = float(x.size)

    # Calculate Sums
    sum_x = np.sum(x)
    sum_x2 = np.sum(np.power(x, 2))
    sum_x3 = np.sum(np.power(x, 3))
    sum_x4 = np.sum(np.power(x, 4))
    sum_y = np.sum(y)
    sum_xy = np.sum(y * x)
    sum_x2y = np.sum(y * np.power(x, 2))

    # intermediate steps
    m = sum_y / n
    den = (sum_x3 * n - sum_x2 * sum_x)
    temp = (
        den * (sum_x * sum_x2 - sum_x3 * n) +
        sum_x2 * sum_x2 * (sum_x * sum_x - n * sum_x2) -
        sum_x4 * n * (sum_x * sum_x - sum_x2 * n)
    )
    temp2 = (
        den * (sum_y * sum_x2 - sum_x2y * n) +
        (sum_xy * n - sum_y * sum_x) * (sum_x4 * n - sum_x2 * sum_x2)
    )

    # calculate curve fit coefficients
    b = temp2 / temp
    a = 1. / den * (
        n * sum_xy -
        sum_y * sum_x -
        b * (sum_x2 * n - sum_x * sum_x)
    )
    c = 1. / n * (sum_y - a * sum_x2 - b * sum_x)

    # calculate sums of squares as well as R^2
    f = a * np.power(x, 2) + b * x + c
    sse = np.sum(np.power(y - f, 2))
    sst = np.sum(np.power(y - m, 2))
    r_squared = 1 - sse / sst

    return a, b, c, r_squared


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
            A cantera gas object used for calculations.
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
                abs(delta_temperature) >
                (error_tol_temperature * guess_temperature)
                or
                abs(delta_velocity) >
                (error_tol_velocity * guess_velocity)
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
            j = derivative_enthalpy_temperature *\
                derivative_pressure_velocity - \
                derivative_pressure_temperature * \
                derivative_enthalpy_velocity
            b = [derivative_pressure_velocity,
                 -derivative_enthalpy_velocity,
                 -derivative_pressure_temperature,
                 derivative_enthalpy_temperature]
            a = [-error_enthalpy,
                 -error_pressure]

            delta_temperature = (b[0] * a[0] + b[1] * a[1]) / j
            delta_velocity = (b[2] * a[0] + b[3] * a[1]) / j

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

    # noinspection SpellCheckingInspection
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
        a = 0
        b = 0
        c = 0

        if use_multiprocessing:
            pool = mp.Pool()

        # Set error tolerances for CJ state calculation
        error_tol_temperature = 1e-4
        error_tol_velocity = 1e-4

        counter = 1
        r_squared = 0.0
        delta_r_squared = 0.0
        adjusted_density_ratio = 0.0

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
            a, b, c, r_squared = cj_curve_fit(
                density_ratio_array,
                cj_velocity_calculations
            )
            adjusted_density_ratio = -b / (2. * a)

            min_density_ratio = adjusted_density_ratio * (1 - 0.001)
            max_density_ratio = adjusted_density_ratio * (1 + 0.001)
            counter += 1

        cj_speed = a * adjusted_density_ratio**2 + \
            b * adjusted_density_ratio + c

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
    # noinspection SpellCheckingInspection
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


class GetError:
    # noinspection SpellCheckingInspection
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

        initial_pressure = initial_state_gas.P
        initial_enthalpy = initial_state_gas.enthalpy_mass
        initial_density = initial_state_gas.density
        initial_velocity = initial_velocity_guess

        working_pressure = working_gas.P
        working_enthalpy = working_gas.enthalpy_mass
        working_density = working_gas.density

        working_velocity = initial_velocity * initial_density / working_density

        sqr_vel_initial = initial_velocity**2
        sqr_vel_working = working_velocity**2

        enthalpy_error = (
                (working_enthalpy + 0.5 * sqr_vel_working) -
                (initial_enthalpy + 0.5 * sqr_vel_initial)
        )

        pressure_error = (
            (
                working_pressure + working_density * sqr_vel_working
            ) - (
                initial_pressure + initial_density * sqr_vel_initial
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
        post_shock_pressure = post_shock_gas.P
        post_shock_enthalpy = post_shock_gas.enthalpy_mass
        post_shock_density = post_shock_gas.density

        working_pressure = working_gas.P
        working_enthalpy = working_gas.enthalpy_mass
        working_density = working_gas.density

        enthalpy_error = (
            working_enthalpy -
            post_shock_enthalpy -
            0.5 * (shock_speed**2)*(
                    (working_density / post_shock_density) + 1
                ) /
            (working_density / post_shock_density - 1)
            )

        pressure_error = (
            working_pressure -
            post_shock_pressure -
            working_density *
            (shock_speed**2) / (
                working_density /
                post_shock_density-1
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
        initial_pressure = initial_state_gas.P
        initial_volume = 1 / initial_state_gas.density

        reflected_pressure = post_shock_gas.P
        reflected_density = post_shock_gas.density
        reflected_volume = 1 / post_shock_gas.density
        reflected_temperature = post_shock_gas.T
        reflected_velocity = np.sqrt(
            (reflected_pressure - initial_pressure) *
            (initial_volume - reflected_volume)
        )

        working_volume = 0.2 / reflected_density
        working_pressure = (
                reflected_pressure +
                reflected_density *
                (incident_shock_speed ** 2) *
                (1 - working_volume / reflected_volume)
        )
        working_temperature = (
                reflected_temperature *
                working_pressure *
                working_volume /
                (reflected_pressure * reflected_volume)
        )

        working_gas.TPX = [
            working_temperature,
            working_pressure,
            post_shock_gas.X
        ]
        working_gas = cls.get_reflected_eq_state(
            reflected_velocity,
            post_shock_gas,
            working_gas
        )
        working_pressure = working_gas.P
        reflected_shock_speed = (
                (working_pressure - reflected_pressure) /
                reflected_velocity /
                reflected_density -
                reflected_velocity
        )

        return [working_pressure, reflected_shock_speed, working_gas]

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
        post_shock_volume = 1 / post_shock_gas.density

        # set reflected guess state
        guess_temperature = working_gas.T
        guess_density = working_gas.density
        guess_volume = 1 / guess_density

        # set initial delta guesses
        delta_temperature = 1000
        delta_volume = 1000

        # equilibrate at guess state
        Properties.equilibrium(
            working_gas,
            guess_density,
            guess_temperature
        )

        # calculate reflected state
        loop_counter = 0
        while (
                (
                    abs(delta_temperature) >
                    error_tol_temperature * guess_temperature
                ) or (
                    abs(delta_volume) >
                    error_tol_specific_volume * guess_volume
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
            delta_temperature = guess_temperature * 0.02
            # equilibrate temperature perturbed state
            Properties.equilibrium(
                working_gas,
                guess_density,
                guess_temperature + delta_temperature
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
                                          err_enthalpy) / delta_temperature
            deriv_pressure_temperature = (err_pressure_perturbed -
                                          err_pressure) / delta_temperature

            # equilibrate working gas with perturbed volume
            delta_volume = 0.02 * guess_volume
            # equilibrate volume perturbed state
            Properties.equilibrium(
                working_gas,
                1 / (guess_volume + delta_volume),
                guess_temperature
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
                                     err_enthalpy) / delta_volume
            deriv_pressure_volume = (err_pressure_perturbed -
                                     err_pressure) / delta_volume

            # solve matrix for temperature and volume deltas
            j = (
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

            delta_temperature = (bb[0] * aa[0] +
                                 bb[1] * aa[1]) / j
            delta_volume = (bb[2] * aa[0] +
                            bb[3] * aa[1]) / j

            # check and limit temperature delta
            delta_temp_max = 0.2 * guess_temperature
            if abs(delta_temperature) > delta_temp_max:
                delta_temperature = (
                        delta_temp_max *
                        delta_temperature /
                        abs(delta_temperature)
                )

            # check and limit specific volume delta
            perturbed_volume = guess_volume + delta_volume
            if perturbed_volume > post_shock_volume:
                delta_volume_max = 0.5 * (post_shock_volume - guess_volume)
            else:
                delta_volume_max = 0.2 * guess_volume

            if abs(delta_volume) > delta_volume_max:
                delta_volume = (
                        delta_volume_max *
                        delta_volume /
                        abs(delta_volume)
                )

            # apply calculated and limited deltas to temperature and spec. vol
            guess_temperature += + delta_temperature
            guess_volume += delta_volume
            guess_density = 1 / guess_volume

            # equilibrate working gas with updated state
            Properties.equilibrium(
                working_gas,
                guess_density,
                guess_temperature
            )

        return working_gas
