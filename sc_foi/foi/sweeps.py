"""
Created on Sun. January 17, 2021

@author: Nick Materise

Summary:
* Plots sweep data from CSV file exported from HFSS
* Sweeps include inductance, capacitance, and geometric parameters

"""

import matplotlib.pyplot as plt
import numpy as np
import post_proc_tools as ppt


def load_eigenmode_sweep_file_data(fname : str, Nmodes : int = 2):
    """
    Loads a CSV file exported from an optimetrics study
    
    Parameters:
    ----------

    fname : str:            path to file
    Nmodes : int:           number of modes to read

    Returns:
    -------

    param : np.ndarray:     parameter from the sweep
    modes:  np.ndarray:     modes as a function of the sweep parameter 

    """
    # Read the data from file
    data = np.genfromtxt(fname, skip_header=1, delimiter=',').T
    
    # Read off the sweep parameter and modes
    param = data[0,:]
    modes = data[1:Nmodes+1,:]
    print(f'param.shape: {param.shape}')
    print(f'modes.shape: {modes.shape}')

    return param, modes


def plot_eigenmodes_vs_param(param : np.ndarray, modes : np.ndarray,
        figname : str, 
        xystrs : dict() = {'xstr' : 'L [nH]', 'ystr' : 'Frequency [GHz]'},
        yscale : float = 1e9,
        xyscales : dict() = {'xscale' : 'linear', 'yscale' : 'linear'},
        figformat : str = 'eps', idx_in : int = 0):
    """
    Plots eigenmodes as a function of a parameter, param

    Parameters:
    ----------

    param : np.ndarray:     parameter data array
    modes : np.ndarray:     eigenmode(s) data array
    xystrs : dict:          dictionary of xy strings for axes labels
    
    """
    # Setup the figure
    fsize = 20
    fig, ax = ppt.init_subplots(fsize, tight_layout=False)
    
    # Plot the results
    if modes.ndim > 1:
        for idx, m in enumerate(modes):
            ax.plot(param, m  / yscale, '.-', label=r'$f_{%d}$' % (idx+1))
    else:
        ax.plot(param, modes  / yscale, '.-', label=r'$f_{%d}$' % (idx_in+1))
    ax.set_xlabel(xystrs['xstr'], fontsize=fsize)
    ax.set_ylabel(xystrs['ystr'], fontsize=fsize)
    ax.set_xscale(xyscales['xscale'])
    ax.set_yscale(xyscales['yscale'])
    leg = ppt.set_leg_outside(ax, fsize)
    plt.tight_layout()
    plt.show()

    # Save the figure to file
    ppt.write_fig_to_file(fig, figname, leg=leg, format=figformat)
