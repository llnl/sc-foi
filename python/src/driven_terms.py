"""
Compute the weights for the terms in the Jacobi-Anger expansion
to determine the expected strength of each term activated by
resonance, harmonic, and subharmonic driving
"""

import numpy as np
import scipy.special

from post_proc_tools import MPLPlotWrapper


class ParametricRates:
    """
    Computes the rates of parametric processes and other multiphoton processes
    that occur at higher levels
    """
    def __init__(self, eps, ws, *args, **kwargs):
        """
        Parameters:
        ----------

        eps:    :float: drive strength in Hz
        ws:     :np.ndarray: resonance frequencies of a collection of modes

        """
        # Implemented valid parametric processes
        self.valid_processes = ['beam_splitter', 'two_mode_squeezing',
                                'one_mode_squeezing']
        self.debug = False

        # Add the inputs to class constructor as class members with the same
        # names as the args or kwargs
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_nth_photon_coefficient(self, n : int, wsb : float, 
                                   process : str) -> float:
        """
        Computes the n-th photon coefficient for the general parametric process
        and the weights for associated error terms
        """
        # Accumulate the terms that satisfy the sum rule
        term = 0.
        errs = 0.
        x = self.eps / (2 * wsb)
        if self.debug:
            print('\n------------------------')
            print(f'{n}-th photon process: ')
        for k in range(n // 2 + 2):
            for kp in range(n // 2 + 2):
                if k + kp == n:
                    if self.debug:
                        print(f'Adding term J_{k} J_{kp} ...')
                    Jk = scipy.special.jn(k, x)
                    Jkp = scipy.special.jn(kp, x)
                    term += Jk * Jkp
                    errs += k * Jk * Jkp

        if self.debug:
            print('------------------------')

        # Check for valid processes and output appropriate scalings to the
        # coefficients
        if process in self.valid_processes:
            match process:
                case 'beam_splitter':
                    return term, errs
                case 'two_mode_squeezing':
                    return term, errs
                case 'one_mode_squeezing':
                    return term, errs
                case _:
                    return term, errs
        else:
            raise ValueError(f'Parametric process {process} not recognized.')

    def get_up_to_Nth_photon_coefficients(self, N : int, wsb : float,
                                          process : str) -> np.ndarray:
        """
        Computes n=1, 2, 3, ..., N photon coefficients for a given parametric
        process with base sideband modulation wsb
        """
        return np.asarray([self.get_nth_photon_coefficient(n, wsb/n, process)
                          for n in range(1, N + 1)])


    def plot_up_to_Nth_photon_coefficients(self, N : int, wsb : float,
                                           process : str, coeff_str : str,
										   filename : str):
        """
        Plots the result of the above function
        """
        # Compute the coefficients
        coeffs = self.get_up_to_Nth_photon_coefficients(N, wsb, process)
        
        # Use the default plotting imported from elsewhere
        myplt = MPLPlotWrapper()
        myplt.ax.plot(list(range(1, N + 1)), coeffs[:, 0], marker='o',
                      label='Coefficients')
        myplt.ax.plot(list(range(1, N + 1)), coeffs[:, 1], marker='x',
                      label='Photon\nnumber\ndependnent\nterms')
        myplt.ax.plot(list(range(1, N + 1)), coeffs[:, 0] + coeffs[:, 1], marker='x',
                      label='  Coefficients\n + Photon\n  number\n  dependent  \n  terms')
        myplt.xlabel = r'$n$-th subharmonic term'
        myplt.ylabel = coeff_str
        myplt.yscale = 'log'
        myplt.set_leg_outside(lsize=16)
        myplt.write_fig_to_file(filename)
