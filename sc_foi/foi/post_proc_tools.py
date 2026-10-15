"""
Post processing tools for plotting, viewing states, etc.
"""

import datetime
import subprocess

import h5py as hdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as scsp
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


class MPLPlotWrapper:
    """
    Class that wraps the matplotlib.pyplot with simpler, built-in functions
    for standard plotting and formatting commands
    """
    def __init__(self, *args, **kwargs):
        """
        Class constructor sets the default operations for the class
        """
        # Initialize the fontsizes and the figure, axes class members
        self.fsize          = 20
        self.tight_layout   = True
        self.leg            = None
        self.is_leg_outside = True
        self._xlabel        = ''
        self._ylabel        = ''
        self._xlim          = None
        self._ylim          = None
        self._xscale        = None
        self._yscale        = None
        self.plot           = None

        # Dimensions of the subplots
        self.xdim = 1
        self.ydim = 1

        # Update the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init_subplots()

        # Set the cyclers
        self.get_set_linestyle_cycler()
        self.get_set_alpha_color_cycler()
        self.get_set_marker_cycler()

    def __del__(self):
        """
        Class destructor to cleanup latent memory
        """
        plt.close('all')

    """
    Class properties
    """
    @property
    def xlabel(self):
        return self._xlabel
    @property
    def ylabel(self):
        return self._ylabel
    @property
    def xlim(self):
        return self._xlim
    @property
    def ylim(self):
        return self._ylim
    @property
    def xscale(self):
        return self._xscale
    @property
    def yscale(self):
        return self._yscale

    """
    Property deleters
    """
    @xlabel.deleter
    def xlabel(self):
        del self._xlabel
    @ylabel.deleter
    def ylabel(self):
        del self._ylabel
    @xscale.deleter
    def xlim(self):
        del self._xlim
    @yscale.deleter
    def yscale(self):
        del self._yscale
    @xscale.deleter
    def xscale(self):
        del self._xscale
    @ylim.deleter
    def ylim(self):
        del self._ylim

    """
    Property setters
    """
    @xlabel.setter
    def xlabel(self, xstr, fsize=None):
        ffsize = fsize if fsize is not None else self.fsize
        # Check the dimensions of the subplot
        if self.xdim > 1 or self.ydim > 1:
            for axx in self.ax.flat:
                axx.set(xlabel=xstr)
		    
            for axx in self.ax.flat:
                axx.label_outer() 
        else:
            self.ax.set_xlabel(xstr, fontsize=ffsize)

        self._xlabel = xstr
    @ylabel.setter
    def ylabel(self, ystr, fsize=None):
        ffsize = fsize if fsize is not None else self.fsize
        # Check the dimensions of the subplot
        if self.xdim > 1 or self.ydim > 1:
            for axx in self.ax.flat:
                axx.set(ylabel=ystr)
            for axx in self.ax.flat:
                axx.label_outer() 
        else:
            self.ax.set_ylabel(ystr, fontsize=ffsize)
            self._ylabel = ystr
    @xlim.setter
    def xlim(self, xlims=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if xlims is not None:
                self.ax.set_xlim(xlims)
                self._xlim = xlims
    @ylim.setter
    def ylim(self, ylims=None):
        if self.xdim > 1 or self.ydim > 1:
            ylim_min = np.min([self.ax[i,j].get_ylim()[0] 
                for i in range(self.xdim) for j in range(self.ydim)])
            ylim_max = np.max([self.ax[i,j].get_ylim()[1] 
                for i in range(self.xdim) for j in range(self.ydim)])
            plt.setp(self.ax, ylim=[ylim_min, ylim_max])
        else:
            if ylims is not None:
                self.ax.set_ylim(ylims)
                self._ylim = ylims
    @xscale.setter
    def xscale(self, xscales=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if xscales is not None:
                self.ax.set_xscale(xscales)
                self._xscale = xscales
    @yscale.setter
    def yscale(self, yscales=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if yscales is not None:
                self.ax.set_yscale(yscales)
                self._yscale = yscales

    def init_subplots(self):
        """
        Returns a figure and axes object with the correct size fonts
        """
        # Get the figure, axes objects
        self.fig, self.ax = plt.subplots(self.xdim, self.ydim,
                tight_layout=self.tight_layout)
    
        # Set the ticks on all edges
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    self.ax[i, j].tick_params(bottom=True, top=True, left=True,
                            right=True)
                    self.ax[i, j].tick_params(labelbottom=True, labeltop=False,
                            labelleft=True, labelright=False)
        else:
            self.ax.tick_params(bottom=True, top=True, left=True,
                    right=True)
            self.ax.tick_params(labelbottom=True, labeltop=False,
                    labelleft=True, labelright=False)
    
        # Set the tick label sizes
        self.set_axes_fonts()
        if self.xdim > 1 or self.ydim > 1:
            self.plot = np.array([self.ax[i,j].plot for i in range(self.xdim)
                                for j in range(self.ydim)])
        else:
            self.plot = self.ax.plot


    def get_set_linestyle_cycler(self):
        """
        Returns a linestyle cycler for plotting
        """
    
        # Different types of dashing styles
        self.linestyle_cycle = [
         (0, (1, 10)),
         (0, (1, 1)),
         (0, (1, 1)),
         (0, (5, 10)),
         (0, (5, 5)),
         (0, (5, 1)),
         (0, (3, 10, 1, 10)),
         (0, (3, 5, 1, 5)),
         (0, (3, 1, 1, 1)),
         (0, (3, 5, 1, 5, 1, 5)),
         (0, (3, 10, 1, 10, 1, 10)),
         (0, (3, 1, 1, 1, 1, 1))]
    
        return self.linestyle_cycle
    
     
    def get_set_alpha_color_cycler(self, alpha=0.5):
        """
        Returns color_cycler default with transparency fraction set to alpha
        """
    
        # Get the color cycler as a hex
        color_cycle_hex = plt.rcParams['axes.prop_cycle'].by_key()['color']
        hex2rgb = lambda hx: [int(hx[0:2],16)/256., \
                              int(hx[2:4],16)/256., \
                              int(hx[4:6],16)/256.]
        color_cycle_rgb = [hex2rgb(cc[1:]) for cc in color_cycle_hex]

        self.alpha_color_cycler = [(*cc, alpha) for cc in color_cycle_rgb]
    
        return self.alpha_color_cycler
    
    
    def get_set_marker_cycler(self):
        """
        Returns a marker style cycler for plotting
        """
    
        # Different marker icons
        self.marker_cycle = ['o', 's', 'D', 'x', 'v', '^', '*', '>', '<', 'p']
    
        return self.marker_cycle


    def set_str_axes_labels(self, axis='x'):
        """
        Sets the axes labels for `axis` to the strings in strs list
        """
        # Set the current axes to ax
        plt.sca(self.ax)
    
        # Select the appropriate axis to apply the labels
        if axis == 'x':
            plt.xticks(range(len(strs)), strs, fontsize=self.fsize)
        elif axis == 'y':
            plt.yticks(range(len(strs)), strs, fontsize=self.fsize)
        else:
            raise KeyError(f'axis ({axis}) invalid.')
    
    def set_axes_fonts(self):
        """
        Set axes font sizes because it should be abstracted away
        """
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    for tick in self.ax[i, j].get_xticklabels():
                        tick.set_fontsize(self.fsize)
                    for tick in self.ax[i, j].get_yticklabels():
                        tick.set_fontsize(self.fsize)
        else:
            for tick in self.ax.get_xticklabels():
                tick.set_fontsize(self.fsize)
            for tick in self.ax.get_yticklabels():
                tick.set_fontsize(self.fsize)
    
    def set_axes_ticks(self, tk, axis='x'):
        """
        Set the values displayed on the ticks
        """
        # Set the x ticks, otherwise, y ticks
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    if axis == 'x':
                        self.ax[i, j].set_xticklabels(tk)
                    elif axis == 'y':
                        self.ax[i, j].set_yticklabels(tk)
                    else:
                        raise ValueError('(%s) axis is invalid.')
        else:
            if axis == 'x':
                self.ax.set_xticklabels(tk)
            elif axis == 'y':
                self.ax.set_yticklabels(tk)
            else:
                raise ValueError('(%s) axis is invalid.')
    
    def set_xaxis_rot(self, angle=45):
        """
        Rotate the x-axis labels
        """
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    for tick in self.ax[i, j].get_xticklabels():
                        tick.set_rotation(angle)
        else:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(angle)


    def set_axes_num_format(self, fmt, axis='x'):
        """
        Sets the number format for the x and y axes in a 1D plot
        """
        if axis == 'x':
            self.ax.xaxis.set_major_formatter(
                    mpl.ticker.StrMethodFormatter(fmt))
        elif axis == 'y':
            self.ax.yaxis.set_major_formatter(
                    mpl.ticker.StrMethodFormatter(fmt))
        else:
            raise KeyError(f'axis {axis} not recognized.')
    

    def set_leg_outside(self, lsize=None):
        """
        Sets the legend location outside
        """
        # Set the legend fontsize to the user input or fsize
        fsize = self.fsize if lsize is None else lsize
        
        # Shrink current axis by 20%
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    box = self.ax[i, j].get_position()
                    self.ax[i, j].set_position([box.x0, box.y0, box.width * 0.8,
                        box.height])
                    
                    # Put a legend to the right of the current axis
                    hdls, legs = self.ax[i, j].get_legend_handles_labels()
                    self.leg = self.ax[i, j].legend(hdls, legs,
                    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fsize,
                            framealpha=0.)
        else:

            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            
            # Put a legend to the right of the current axis
            hdls, legs = self.ax.get_legend_handles_labels()
            self.leg = self.ax.legend(hdls, legs, loc='center left',
                    bbox_to_anchor=(1, 0.5), fontsize=fsize, framealpha=0.)

        # Update the legend state
        self.is_leg_outside = True
    
    def set_leg_hdls_lbs(self, lsize=None, loc='best'):
        """
        Set the legend handles and labels
        """
        # Set the legend fontsize to the user input or fsize
        fsize = self.fsize if lsize is None else lsize
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    hdl, leg = self.ax[i, j].get_legend_handles_labels()
                    self.ax[i, j].legend(hdl, leg, loc=loc, fontsize=fsize,
                                         framealpha=0.)
        else:
            hdls, legs = self.ax.get_legend_handles_labels()
            self.leg = self.ax.legend(hdls, legs, loc=loc, fontsize=fsize,
                                      framealpha=0.)

    def write_fig_to_file(self, fname):
        """
        Writes a figure object to file, sets legends accordingly
        """
        fmt = fname.split('.')[-1]
        # Check for no legends
        if self.leg is None:
            self.fig.savefig(fname, format=format, transparent=True)
        
        # Otherwise save with legends
        else:
            ## Check for setting legend outside
            if self.is_leg_outside:
                self.fig.savefig(fname, format=fmt,
                        bbox_extra_artists=(self.leg, ), bbox_inches='tight',
                        transparent=True)
            else:
                ### Check for the ax object to set the legends
                self.fig.savefig(fname, format=fmt, transparent=True)

        print(f'{fname} written to file.')


def get_linestyle_cycler():
    """
    Returns a linestyle cycler for plotting
    """

    # Different types of dashing styles
    linestyle_cycle = [
     (0, (1, 10)),
     (0, (1, 1)),
     (0, (1, 1)),
     (0, (5, 10)),
     (0, (5, 5)),
     (0, (5, 1)),
     (0, (3, 10, 1, 10)),
     (0, (3, 5, 1, 5)),
     (0, (3, 1, 1, 1)),
     (0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 10, 1, 10, 1, 10)),
     (0, (3, 1, 1, 1, 1, 1))]

    return linestyle_cycle

 
def get_alpha_color_cycler(alpha=0.5):
    """
    Returns color_cycler default with transparency fraction set to alpha
    """

    # Get the color cycler as a hex
    color_cycle_hex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hex2rgb = lambda hx: [int(hx[0:2],16)/256., \
                          int(hx[2:4],16)/256., \
                          int(hx[4:6],16)/256.]
    color_cycle_rgb = [hex2rgb(cc[1:]) for cc in color_cycle_hex]

    return [(*cc, alpha) for cc in color_cycle_rgb]


def get_marker_cycler():
    """
    Returns a marker style cycler for plotting
    """

    # Different marker icons
    marker_cycle = ['D', 'x', 'o', 's', 'v', '^', '*', '>', '<', 'p']

    return marker_cycle



def set_axes_fonts(ax, fsize):
    """
    Set axes font sizes because it should be abstracted away
    """
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(fsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(fsize)


def set_axes_ticks(ax, tk, axis='x'):
    """
    Set the values displayed on the ticks
    """

    # Set the x ticks, otherwise, y ticks
    if axis == 'x':
        ax.set_xticklabels(tk)
    elif axis == 'y':
        ax.set_yticklabels(tk)
    else:
        raise ValueError('(%s) axis is invalid.')
    

def set_xaxis_rot(ax, angle=45):
    """
    Rotate the x-axis labels
    """
        
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)


def init_subplots(fsize=20, tight_layout=True):
    """
    Returns a figure and axes object with the correct size fonts
    """

    # Get the figure, axes objects
    fig, ax = plt.subplots(1, 1, tight_layout=tight_layout)

    # Set the ticks on all edges
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, \
                   labelleft=True, labelright=False)

    # Set the tick label sizes
    set_axes_fonts(ax, fsize)
    

    return fig, ax


def set_axes_num_format(ax, fmt):
    """
    Sets the number format for the x and y axes in a 1D plot
    """
    
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(fmt))
    # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(fmt))


def set_leg_outside(ax, fsize):
    """
    Sets the legend location outside
    """
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    hdls, legs = ax.get_legend_handles_labels()
    leg = ax.legend(hdls, legs, loc='center left', bbox_to_anchor=(1, 0.5), \
                    fontsize=fsize, framealpha=0.)

    return leg


def set_leg_hdls_lbs(ax, fsize, loc='best'):
    """
    Set the legend handles and labels
    """

    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc=loc, fontsize=fsize, framealpha=0.)


def stderr_fill(ax, xval, yval, yerr, fill_color, alpha=0.5):
    """
    Shaded region for standard deviation on a linear plot
    """

    # Shaded region command
    ax.fill_between(xval, yval + yerr, yval - yerr, \
            facecolor=fill_color, alpha=alpha)


def write_fig_to_file(fig, fname, leg=None, ax=None,\
                      is_leg_outside=True, \
                      format='pdf', fsize=20, \
                      leg_loc='best'):
    """
    Writes a figure object to file, sets legends accordingly
    
    Parameters:
    ----------

    fig:                    matplotlib figure object
    fname:                  path to output file 
    leg:                   matplotlib legends object
    ax:                     matplotlib axes object
    is_leg_outside:         True if legends set outside of figure, else false
    format:                 output file format
    fsize:                  legend fontsize
    leg_loc:                legend location

    """

    # Check for no legends
    if leg is None:
        fig.savefig(fname, format=format, transparent=True)
    
    # Otherwise save with legends
    else:
        ## Check for setting legend outside
        if is_leg_outside:
            fig.savefig(fname, format=format, \
                  bbox_extra_artists=(leg, ), bbox_inches='tight', \
                  transparent=True)
        else:
            ### Check for the ax object to set the legends
            if ax is None:
                pass
            else:
                set_leg_hdls_lbs(ax, fsize, loc=leg_loc)
            
            fig.savefig(fname, format=format, transparent=True)



def plot_2d_cmap(x, y, z, fname,
                 xstr='', ystr='',
                 tstr='', cbar_str='',
                 cmap=cm.inferno):
    """
    Plot 2D colormap data such that 

         -----------------------
         |                     |
         |                     |
    y    |          z          |
         |                     |
         |                     |
         ----------------------- 
    
                    x
    
    Parameters:
    ----------

    x, y:       independent variables
    z:          resulting data, z = z(x, y) 
    fname:      output figure filename, relative path with file extension
    xstr:       x-axis label
    ystr:       y-axis label
    tstr:       title label
    cbar_str:   lable for color bar
    cmap:       colormap to use on the 2D figure output

    """

    # Setup the color map, normalizations, etc
    norm = mpl.colors.Normalize(z.min(), z.max())

    # Fontsize
    fsize = 24; tsize = 26;

    # Setup the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
    levels = MaxNLocator(nbins=20).tick_values(z.min(), z.max())
    plt1 = ax.contourf(x, y, z, 100, cmap=cmap, norm=norm, levels=levels)
    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)
    ax.set_title(tstr, fontsize=tsize)

    # Set the axis tick labels to a reasonable size
    set_axes_fonts(ax, fsize)
    
    # Set the color bar, offset the title slightly from top
    cbar = fig.colorbar(plt1, ax=ax)
    cbar.ax.set_title(cbar_str, fontsize=fsize, y=1.025)
    cbar.ax.tick_params(labelsize=fsize)

    # Write the results to file
    fig.savefig(fname, format='pdf', transparent=True) 
        

def gnuplot_term_trace(x, y, title=None):
    """
    Uses gnuplot to plot data in the terminal
    """

    # Open a process and pipe commands to gnuplot
    gnuplot = subprocess.Popen(["/usr/bin/gnuplot"], stdin=subprocess.PIPE)

    # Set the terminal to dumb
    gnuplot.stdin.write(b"set term dumb 79 25\n")
    
    # Plot the data, starting with the lines command
    tstr = str.encode(title) if title != None else b"Line1"
    gnuplot.stdin.write(b"plot '-' using 1:2 title '%b' with dots\n" % tstr)
    
    # Iterate over the input data
    for i, j in zip(x, y):
        gnuplot.stdin.write(b"%f %f\n" % (i, j))

    # Write the execute command, then flush 
    gnuplot.stdin.write(b"e\n")
    gnuplot.stdin.flush()
    gnuplot.stdin.write(b"quit")


def gnuplot_dumb_traces(xlist, ylist, 
                        tlist=None, linespoints='dots',
                        xlim=[], ylim=[], xyscales=['linear', 'linear']):
    """
    Plot multiple traces, each input is assumed a list
    """
    # Open a process and pipe commands to gnuplot
    print('\n')
    gnuplot = subprocess.Popen(["/usr/bin/gnuplot"], stdin=subprocess.PIPE)

    # Set the terminal to dumb
    gnuplot.stdin.write(b"set term dumb 79 25\n")

    # Set the x and y ranges if not empty
    if xlim != []:
        print(f'Setting xlim to {xlim} ...')
        xrange_str = 'set xrange [%g:%g]\n' % (xlim[0], xlim[1])
        gnuplot.stdin.write(str.encode(xrange_str))

    # Set the x and y ranges if not empty
    if ylim != []:
        print(f'Setting ylim to {ylim} ...')
        yrange_str = 'set yrange [%g:%g]\n' % (ylim[0], ylim[1])
        gnuplot.stdin.write(str.encode(yrange_str))

    # Get the number of entries
    Nitems = len(xlist)

    # Set the linespoints option
    lp = str.encode(linespoints)

    # Set the title strings
    tstrs = [str.encode(tt) for tt in tlist] if tlist != None \
            else [b'Line%b' % (str.encode('%d' % i)) for i in range(Nitems)]

    # Set the x and y scales
    xscale_str = 'set %sscale x\n' % xyscales[0] \
                    if xyscales[0] == 'log' else 'unset logscale x\n'
    gnuplot.stdin.write(str.encode(xscale_str))
    yscale_str = 'set %sscale y\n' % xyscales[1] \
                    if xyscales[1] == 'log' else 'unset logscale y\n'
    gnuplot.stdin.write(str.encode(yscale_str))

    # Set the titles in one shot
    plot_str = b"plot %b" % (b",".join([b"'-' u 1:2 t '%b' w %b" \
                % (tt, lp) for tt in tstrs]))
    plot_str += b"\n"
    gnuplot.stdin.write(plot_str) 

    # Iterate over the inputs
    for x, y in zip(xlist, ylist):
    
        # Iterate over the input data
        for i, j in zip(x, y):
            gnuplot.stdin.write(b"%.16f %.16f\n" % (i, j))

        # Write the execute command, then flush 
        gnuplot.stdin.write(b"e\n")

    # Flush gnuplot to view
    gnuplot.stdin.flush()
    gnuplot.stdin.write(b"quit")
    print('\n')


def gnuplot_dumb_boxes(x, y, title=None):
    """
    Plot a single bar chart with x, y data
    """

    # Open a process and pipe commands to gnuplot
    gnuplot = subprocess.Popen(["/usr/bin/gnuplot"], stdin=subprocess.PIPE)

    # Set the terminal to dumb
    gnuplot.stdin.write(b"set term dumb 79 25\n")
    
    # Plot the data, starting with the lines command
    gnuplot.stdin.write(b"plot '-' using 1:2 t '' w boxes\n")
    
    # Iterate over the input data
    for i, j in zip(x, y):
        gnuplot.stdin.write(b"%f %f\n" % (i, j))

    # Write the execute command, then flush 
    gnuplot.stdin.write(b"e\n")
    if title != None:
        print('\n%s\n' % title)
    gnuplot.stdin.flush()
    gnuplot.stdin.write(b"quit")


def plot_expect(tpts, op_avg, op_name='',
                tscale='ns', file_ext=None,
                plot_phase=False, ms=None):
    """
    Plot the expectation value of an operator as a function of time
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    fsize = 24; tsize = 26;
    set_axes_fonts(ax, fsize)
    ax.plot(tpts, np.abs(op_avg), marker=ms)
    
    # Set the axes labels
    xstr = 'Time (%s)' % tscale
    ystr = r'$\langle{%s}\rangle$' % op_name

    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)
    
    # Save the figure to file
    if file_ext is not None:
        fig.savefig('figs/expect_%s.pdf' % file_ext, format='pdf', \
                transparent=True) 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        fig.savefig('figs/expect_%s.pdf' % tstamp, format='pdf', \
                transparent=True) 


def plot_expect_complex_ab(op_a, op_b, 
                            opname, snames, 
                            fext=None, scale=1):
    """
    Generates the quadrature plot (Im<op> vs. Re<op>) in states |a>, |b>
    
    Parameters:
    ----------

    op_a, op_b:     two operators' expectation values as functions of time 
                    for states a, b 
    opnames:        operator name 
    snames:         names of states corresponding to op_a, op_b
    scale:          amount to divide real and imaginary components by

    
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
            tight_layout=True)
    fsize = 24; tsize = 26; lw = 1.5; lsize=20
    set_axes_fonts(ax, fsize)

    # Convert the input to numpy arrays
    if op_a.__class__ != np.ndarray:
        op_a = np.array(op_a)
    if op_b.__class__ != np.ndarray:
        op_b = np.array(op_b)

    ax.plot(op_a.real/scale, op_a.imag/scale,
            'ro-', linewidth=lw,
            label=r'$\left|{%s}\right>$' % snames[0])
    ax.plot(op_b.real/scale, op_b.imag/scale,
            'bo-', linewidth=lw,
            label=r'$\left|{%s}\right>$' % snames[1])

    # Set the x/y limits
    amax = op_a.max(); bmax = op_b.max()

    # Set the axes labels
    xstr = r'$\Re\langle{%s}\rangle$' % opname
    ystr = r'$\Im\langle{%s}\rangle$' % opname
    ax.set_xlabel(xstr, fontsize=fsize)
    ax.set_ylabel(ystr, fontsize=fsize)

    # Set the legends
    hdls, legs = ax.get_legend_handles_labels()
    ax.legend(hdls, legs, loc='best', fontsize=lsize)
    
    # Save the figure to file
    if fext is not None:
        print('Writing figure to figs/%s_expect_%s.pdf' % (opname, fext))
        fig.savefig('figs/%s_expect_%s.pdf' % (opname, fext), format='pdf'\
                , transparent=True) 
    else:
        tstamp = datetime.datetime.today().strftime('%y%m%d_%H:%M:%S')
        print('Writing figure to figs/%s_expect_%s_%s.pdf' \
                % (opname, fext, tstamp))
        fig.savefig('figs/%s_expect_%s_%s.pdf' % (opname, fext, tstamp),
                format='pdf', transparent=True) 
    

def plot_liouvillian_map(L, fext='', cmap=cm.inferno, use_sparse=False):
    """
    Generates a color map of the Liouvillian for a particular system
    """

    # Generate the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)

    # Set the figure axes
    fsize = 24

    # Generate the color map
    if use_sparse:
        Ls = scsp.csc_matrix(L)
        print('Number of non-zero elements in L: %d' % Ls.data.size)
        ax.spy(Ls)
        set_axes_fonts(ax, fsize)
        fext = fext + '_sparse'
    else:
        ax.imshow(L, cmap=cmap)
    fig.savefig('figs/liouvillian_cmap_%s.pdf' % fext, format='pdf', \
            transparent=True)


def plot_hamiltonian_map(H, fext='', cmap=cm.inferno, use_sparse=False):
    """
    Generates a color map of the Hamiltonian for a particular system
    """

    # Generate the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Set the figure axes
    fsize = 24

    # Generate the color map
    if use_sparse:
        ax.spy(scsp.csc_matrix(H))
        fext = fext + '_sparse'
        set_axes_fonts(ax, fsize)
    else:
        ax.imshow(H, cmap=cmap)
    fig.savefig('figs/hamiltonian_cmap_%s.pdf' % fext, format='pdf', \
            transparent=True)


def plot_post_qeng(fprefix, tscale='us'):
    """
    Plot the results of the quantum heat engine by reading the file
    """

    # Read the hdf5 file
    fid_mbl = hdf.File('data/mbl_%s.hdf5' % fprefix, 'r')
    fid_eth = hdf.File('data/eth_%s.hdf5' % fprefix, 'r')

    print(f'fid_mbl.keys(): {fid_mbl.keys()}')
    print(f'fid_eth.keys(): {fid_eth.keys()}')

    # Get the fidelities, residual energies
    tpts_exp  = fid_eth['tpts_exp'][()]
    tpts_comp = fid_mbl['tpts_comp'][()]
    fid1_eth  = fid_eth['fid1_eth'][()]
    fid0_mbl  = fid_mbl['fid0_mbl'][()]
    E1eth_res = fid_eth['E1eth_res'][()]
    E0mbl_res = fid_mbl['E0mbl_res'][()]

    # Get the times in us
    texp = tpts_exp.max()
    tcomp = tpts_comp.max()

    # Plot the results on separate figures
    ## Fidelity of psi_mbl,1 -> psi_eth,1
    print('Plotting fidelities ...')
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    fsize = 22
    fmt = '{x:3.2f}'
    set_axes_fonts(ax1, fsize)
    set_axes_num_format(ax1, fmt)
    set_xaxis_rot(ax1)
    ax1.plot(tpts_exp, fid1_eth)
    ax1.set_xlabel(r'Time [%s]' % tscale, fontsize=fsize)
    ax1.set_ylabel('Fidelity', fontsize=fsize)
    ax1.set_title(r'Fidelity \
    $\left<\psi_{\mathrm{MBL, 1}}\right|\left.\psi_{\mathrm{ETH, 1}}\right>$',\
                fontsize=fsize)
    fig1.savefig('figs/fid1_eth_%0.2fus.pdf' % texp, format='pdf', \
            transparent=True)

    ## Fidelity of psi_eth,0 -> psi_mbl,0
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    set_axes_fonts(ax2, fsize)
    set_axes_num_format(ax2, fmt)
    set_xaxis_rot(ax2)
    ax2.plot(tpts_comp, fid0_mbl)
    ax2.set_xlabel('Time [%s]' % tscale, fontsize=fsize)
    ax2.set_ylabel('Fidelity', fontsize=fsize)
    ax2.set_title(r'Fidelity \
    $\left<\psi_{\mathrm{ETH, 0}}\right|\left.\psi_{\mathrm{MBL, 0}}\right>$',\
                fontsize=fsize)
    fig2.savefig('figs/fid0_mbl_%0.2fus.pdf' % tcomp, format='pdf', \
            transparent=True)

    ## Residual energy from psi_mbl,1 -> psi_eth,1
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    set_axes_fonts(ax3, fsize)
    set_axes_num_format(ax3, fmt)
    set_xaxis_rot(ax3)
    ax3.plot(tpts_exp, E1eth_res)
    ax3.set_xlabel('Time [%s]' % tscale, fontsize=fsize)
    ax3.set_ylabel('Residual Energy', fontsize=fsize)
    ax3.set_title(r'Residual Energy from \
    $\left|\psi_{\mathrm{MBL,1}}\right>\to\left|\psi_{\mathrm{ETH,1}}\right>$',\
                fontsize=fsize)
    fig3.savefig('figs/E1eth_res_%0.2fus.pdf' % texp, format='pdf', \
            transparent=True)

    ## Residual energy from psi_eth,0-> psi_mbl,0
    print('Plotting residual energies ...')
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    set_axes_fonts(ax4, fsize)
    set_axes_num_format(ax4, fmt)
    set_xaxis_rot(ax4)
    ax4.plot(tpts_comp, E0mbl_res)
    ax4.set_xlabel('Time [%s]' % tscale, fontsize=fsize)
    ax4.set_ylabel('Residual Energy', fontsize=fsize)
    ax4.set_title(r'Residual Energy from \
    $\left|\psi_{\mathrm{ETH,0}}\right>\to\left|\psi_{\mathrm{MBL,0}}\right>$',\
                fontsize=fsize)
    fig4.savefig('figs/E0mbl_res_%0.2fus.pdf' % tcomp, format='pdf', \
            transparent=True)


def write_to_hdf(data_list, key_list, fname):
    """
    Writes a list of arrays to data sets with names in key_list to an hdf5 at
    the path given by fname
    
    Parameters:
    ----------

    data_list:      list of arrays to write to data sets
    key_list:       list of string keys for each data set in data_list in the
                    same order as data_list 
    fname:          path to the hdf5 file output

    """

    # Open the new hdf5 file and get the file handle
    fid = hdf.File(fname, 'w')

    # Iterate over the data_list and key_list
    for d, k in zip(data_list, key_list):
        # print('{}: {}'.format(k, d))
        d = np.asarray(d)
        fid.create_dataset(k, data=d)

    # Close the file handle
    fid.close()


def read_from_hdf(key_list, fname):
    """
    Reads a list of data sets given in the key_list and returns the value arrays
    associated with those keys
    """

    # Open the hdf file for reading
    fid = hdf.File(fname, 'r')

    # Initialize the output list
    dout = []

    # Iterate over all of the keys passed as key_list
    for idx, k in enumerate(key_list):

        ## XXX: This is poor memory management, fix later by calling a native
        ## HDF5 reading function that returns a list of data sets or something
        ## equivalent that is more memory efficient
        dout.append(fid[k][()])

    return dout


def combine_hdf_chkpts(chkpt_fnames, key_list, fname_out):
    """
    Reads data from checkout points and writes results to single file
    """
    # Create dictionary for output data
    ddict = {'%s' % k : [] for k in key_list}

    # Iterate over all check point file names
    for fname in chkpt_fnames:
        print('Reading checkpoint file (%s) ...' % fname)

        ## Read the data from file
        chkpt_data = read_from_hdf(fname, key_list)
        
        ## Iterate over keys and accumulate the data
        for kidx, k in enumerate(key_list):
            ddict[k].append(chkpt_data[kidx])

    # Write the data to file
    print('Writing combined file to (%s) ...' % fname_out)
    write_to_hdf([ddict[k] for k in key_list], key_list, fname_out)


def post_plot_fid_Eres_from_hdf(fname, keys, Ns, use_twinx=True, use_std=False,
                                Nreal=1, fext='', plot_occ=False,
                                axescales=['linear', 'log']):
    """
    Given a filename and list of keys, plot the data from the keys
    
    Parameters:
    ----------

    fname:      path to the hdf5 file with the data to plot 
    keys:       keys of the ordinates and abscissa to plot, e.g.
                ['fid1', 'Eres1'] -- ['da_out'] for trajectories of both
                quantities
    Ns:         number of sites in the lattice simulation
    use_twinx:  plot on same figure with different y-axes
    use_std:    highlight the standard deviation
    Nreal:      1 / number of realizations for plotting the standard deviation
                or plotting the standard error

    """

    # Get the hdf5 file handle
    fid = hdf.File(fname, 'r')

    # Get the times to plot against
    tpts = fid['tpts'][()]

    # Font settings
    fsize = 20

    # Plot the data on the same set of axes
    if use_twinx:

        # Create a separate figure for each key
        fig, ax1 = plt.subplots(1, 1, tight_layout=True)
        
        # Set the axes fontsizes
        set_axes_fonts(ax1, fsize)

        # Get the ylabel from the key
        ystr2 = ' '.join(keys[0].split('_'))
        ystr1 = ' '.join(keys[1].split('_'))

        # Set the colors for each line
        color2 = 'tab:blue'
        color1 = 'tab:red'

        # Plot the result of the key
        ax1.plot(tpts, fid[keys[1]][()], color=color1)
        ax1.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
        ax1.set_ylabel(ystr1, fontsize=fsize, color=color1)

        # Rotate the x-axis labels
        # set_xaxis_rot(ax1)

        # Set the axis color
        ax1.tick_params(axis='y', labelcolor=color1)

        # Get the twin-axis
        ax2 = ax1.twinx()
        set_axes_fonts(ax2, fsize)
        ax2.plot(tpts, fid[keys[0]][()], color=color2)
        ax2.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
        ax2.set_ylabel(ystr2, fontsize=fsize, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([-0.1, 1.1])
        ax1.set_ylim([0, 340])
        
        ## Add filling for standard deviation
        if use_std:

            # Reduce the fluctuations by N  
            N = Nreal

            ## Shading for Fidelity
            ax2.fill_between(tpts, \
                    fid[keys[0]][()] \
                    +fid[keys[0]+' Std'][()]/np.sqrt(N), \
                             fid[keys[0]][()] \
                             -fid[keys[0]+' Std'][()]/np.sqrt(N), \
                             facecolor=color2, alpha=0.5)

            ## Shading for Residual Energy
            ax1.fill_between(tpts, \
                    fid[keys[1]][()] \
                    +fid[keys[1]+' Std'][()]/np.sqrt(N), \
                    fid[keys[1]][()] \
                    -fid[keys[1]+' Std'][()]/np.sqrt(N), \
                             facecolor=color1, alpha=0.5)
        
        # Save and close the figure
        fstr = '_'.join(fname.split('.')[0:-1])\
                +'_'+keys[0]+'_'+'_'.join(keys[1].split(' '))\
                +'.pdf' if fext.count('') < 2 \
                else \
                '_'.join(fname.split('.')[0:-1])\
                +'_'+keys[0]+'_'+'_'.join(keys[1].split(' '))\
                +'_'+fext+'.pdf'

        fig.savefig(fstr, format='pdf', transparent=True)
        plt.close()

    # Otherwise, produce different figures
    else:

        # Get the fidelities and the residual energies
        for k in keys:

            # Create a separate figure for each key
            fig, ax = plt.subplots(1, 1, tight_layout=True)
            
            # Set the axes fontsizes
            set_axes_fonts(ax, fsize)

            # Rotate the x-axis labels
            # set_xaxis_rot(ax)

            # Get the ylabel from the key
            ystr = ' '.join(k.split('_'))

            # Plot the result of the key
            if sum(fid[k][()] < 0) > 1:
                fid_data = np.abs(fid[k][()])
            else:
                fid_data = fid[k][()]
            ax.plot(tpts, fid_data, '.-')
            ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
            ax.set_ylabel(ystr, fontsize=fsize)
            ax.set_xscale(axescales[0])
            ax.set_yscale(axescales[1])
        
            # Save and close the figure
            fstr = '_'.join(fname.split('.')[0:-1])\
                       +'_'+k+'.pdf' if fext.count('') < 2 \
                    else \
                    '_'.join(fname.split('.')[0:-1])\
                       +'_'+k+'_'+fext+'.pdf'
            fig.savefig(fstr, format='pdf', transparent=True)
            plt.close()

    # Plot of the color map of occupations
    ## Get the occupation data from the hdf file
    if plot_occ:
        occ = fid['Occupations'][()]
        Nslist = np.linspace(1, Ns, Ns)
        
        ## Call the cmap plotter
        cmp_fstr = '_'.join(fname.split('.')[0:-1])\
                   +'_occ_'+str(Ns)+'Ns.pdf'
        plot_2d_cmap(tpts, Nslist, occ.real, cmp_fstr, \
                r'Time[$\mu$s]', 'Sites', \
                tstr='Site Occupations', cbar_str='Occ. No.')

    # Close the hdf5 file handle
    fid.close()


def post_plot_fid_Eres_real_from_hdf(fname, keys, use_twinx=True,
                                     use_std=False, Nreal=1, fext=''):
    """
    Given a filename and list of keys, plot the individual realization data from
    the keys
    
    Parameters:
    ----------

    fname:      path to the hdf5 file with the data to plot 
    keys:       keys of the ordinates and abscissa to plot, e.g.
                ['fid1', 'Eres1'] -- ['da_out'] for trajectories of both
                quantities
    use_twinx:  plot on same figure with different y-axes
    use_std:    highlight the standard deviation
    Nreal:      1 / number of realizations for plotting the standard deviation
                or plotting the standard error

    """

    # Get the hdf5 file handle
    fid = hdf.File(fname, 'r')

    # Get the times to plot against
    tpts = fid['tpts'][()]

    # Font settings
    fsize = 20

    # Read the trajectory data
    da_out = fid['da_out'][()]

    ## Get the residual energies and fidelities separately
    fid_traj  = np.array([da_out[i][0] for i in range(Nreal)])
    Eres_traj = np.array([da_out[i][1] for i in range(Nreal)])
    ddict = {'Fidelity' : fid_traj, 'Residual Energy' : Eres_traj}
    

    # Get the fidelities and the residual energies
    for k in keys:

        # Create a separate figure for each key
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        
        # Set the axes fontsizes
        set_axes_fonts(ax, fsize)

        # Rotate the x-axis labels
        set_xaxis_rot(ax)

        # Get the ylabel from the key
        ystr = ' '.join(k.split('_'))

        # Plot the result of the key
        for d in ddict[k]:
            ax.plot(tpts, d, alpha=0.25, lw=1)
        ax.plot(tpts, fid[k][()], 'k--', lw=2)
        ax.set_xlabel(r'Time [$\mu$s]', fontsize=fsize)
        ax.set_ylabel(ystr, fontsize=fsize)
    
        # Save and close the figure
        fstr = '_'.join(fname.split('.')[0:-1])\
                +'_'+'_'.join(k.split(' '))\
                +'_real.pdf' if fext.count('') < 2 \
                else \
                '_'.join(fname.split('.')[0:-1])\
                +'_'+'_'.join(k.split(' '))\
                +'_'+fext+'_real.pdf'
        fig.savefig(fstr, format='pdf', transparent=True)
        plt.close()


    # Close the hdf5 file handle
    fid.close()


def twinx_plot(xlist, ylist, xstr, ystrs, fname, use_same_yscale=False):
    """
    Plots data (x1, y1) and (x2, y2) on shared x-axis using differnt y-axes
    
    Parameters:
    ----------

    xlist:      [x1_array, x2_array]
    ylist:      [y1_array, y2_array]
    xstr:       x-axis label
    ystrs:      ['yleft-label', 'yright-label']
    fname:      output file name

    """
    # Default font sizes
    fsize = 20
    
    # Create a separate figure for each key
    fig, ax1 = plt.subplots(1, 1, tight_layout=True)
    
    # Set the axes fontsizes
    set_axes_fonts(ax1, fsize)
    
    # Set the colors for each line
    color1 = 'tab:blue'
    color2 = 'tab:red'
    
    # Plot the result of the key
    ax1.plot(xlist[0], ylist[0], color=color1)
    ax1.set_xlabel(xstr, fontsize=fsize)
    ax1.set_ylabel(ystrs[0], fontsize=fsize, color=color1)
    
    # Rotate the x-axis labels
    set_xaxis_rot(ax1)
    
    # Set the axis color
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Get the twin-axis
    ax2 = ax1.twinx()
    set_axes_fonts(ax2, fsize)
    ax2.plot(xlist[1], ylist[1], color=color2)
    ax2.set_xlabel(xstr, fontsize=fsize)
    ax2.set_ylabel(ystrs[1], fontsize=fsize, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set the y-axes to have the same limits
    if use_same_yscale:
        ymax = 1.1 * max(ylist[0].max(), ylist[1].max())
        ymin = min(ylist[0].min(), ylist[1].min())
        ax1.set_ylim([ymin, ymax])
        ax2.set_ylim([ymin, ymax])
        ax1.set_xlim([xlist[0].min(), xlist[0].max()])
        ax2.set_xlim([xlist[1].min(), xlist[1].max()])

    # Write the figure to file
    # Assuming the format is the same as the file extension
    fig.savefig(fname, format=fname.split('.')[-1], transparent=True)


def twiny_plot(xlist, ylist, xstrs, ystr, fname, fsize=20, \
               axes_scales=['linear', 'linear']):
    """
    Plots data (x1, y1) and (x2, y2) on shared y-axis using differnt x-axes
    
    Parameters:
    ----------

    xlist:          [x1_array, x2_array]
    ylist:          [y1_array, y2_array]
    xstr:           ['xleft-label', 'xright-label']
    ystrs:          y-axis label
    fname:          output file name
    fsize:          font size
    axes_scales:    scales for [x-axis, y-axis]    

    """
    # Create a separate figure for each key
    fig, ax1 = plt.subplots(1, 1, tight_layout=True)
    
    # Set the axes fontsizes
    set_axes_fonts(ax1, fsize)
    
    # Set the colors for each line
    color1 = 'tab:blue'
    color2 = 'tab:red'

    # Check for two items in the list
    if (len(xlist) > 1) and (len(ylist) > 1):
    
        # Plot the result of the key
        ax1.plot(xlist[0], ylist[0], '.-', color=color1)
        ax1.set_xlabel(xstrs[0], fontsize=fsize)
        ax1.set_ylabel(ystr, fontsize=fsize, color=color1)
        
        # Set the axis color
        ax1.tick_params(axis='x', labelcolor=color1)
        ax1.set_xscale(axes_scales[0])
        ax1.set_yscale(axes_scales[1])
        
        # Get the twin-axis
        ax2 = ax1.twiny()
        set_axes_fonts(ax2, fsize)
        ax2.plot(xlist[1], ylist[1], '.-', color=color2)
        ax2.set_xlabel(xstrs[1], fontsize=fsize)
        ax2.set_ylabel(ystr, fontsize=fsize, color=color2)
        ax2.tick_params(axis='x', labelcolor=color2)
        ax2.set_xscale(axes_scales[0])
        ax2.set_yscale(axes_scales[1])

    # Check for one item in list, only plot that item on both axes
    elif (len(xlist) > 1) and (len(ylist) == 1):

        # Plot the result of the key
        ax1.plot(xlist[0], ylist[0], '.-', color=color1)
        ax1.set_xlabel(xstrs[0], fontsize=fsize)
        ax1.set_ylabel(ystr, fontsize=fsize)
        ax1.set_xscale(axes_scales[0])
        ax1.set_yscale(axes_scales[1])
        
        # Get the twin-axis
        ax2 = ax1.twiny()
        set_axes_fonts(ax2, fsize)
        ax2.plot(xlist[1], ylist[0], '.-', color=color1)
        ax2.set_xlabel(xstrs[1], fontsize=fsize)
        ax2.set_ylabel(ystr, fontsize=fsize)
        ax2.set_xscale(axes_scales[0])
        ax2.set_yscale(axes_scales[1])

    # Raise exception for zero length
    else:
        raise ValueError('len(x): %d and len(y): %d not supported' \
                % (len(x), len(y)))

    # Write the figure to file
    # Assuming the format is the same as the file extension
    fig.savefig(fname, format=fname.split('.')[-1], transparent=True)


def plot_eres_density(Nslist, EresNs, fname_out, Tlist, use_leg=False):
    """
    Plots the residual energy density
    
    Parameters:
    ----------

    Nslist:         list of number of sites 
    EresNs:         residual energy / Ns_i
    fname_out:      output filename for the figure
    Tlist:          list of adiabatic evolution times

    """

    # Default font sizes
    fsize = 20
    lsize = 22
    
    # Create a separate figure for each key
    fig, ax1 = plt.subplots(1, 1) #, tight_layout=True)
    
    # Set the axes fontsizes
    set_axes_fonts(ax1, fsize)

    # Iterate over the times
    for idx, T in enumerate(Tlist):

        # Plot the data
        ax1.plot(Nslist, EresNs[idx], '*-', label=r'T=%3.1f $\mu$s' % T)
        ax1.set_xlabel(r'Number of Sites $(L)$', fontsize=fsize)
        ax1.set_ylabel(r'$\left|E - \left<H\right>\right|$ / $L$', \
                       fontsize=fsize)

    # Set title
    ax1.set_title(r'Residual Energy Density', fontsize=fsize)
    ax1.set_ylim([-0.1, 5])

    # Set the legend
    if use_leg:
        leg = set_leg_outside(ax1, fsize)

        # Save the results to file
        fig.savefig(fname_out, format='pdf', \
                bbox_extra_artists=(leg, ), bbox_inches='tight', \
                transparent=True)

    else:
        # Set aspect ratio to 1
        ax1.set_aspect(1)

        # Save the results to file
        fig.savefig(fname_out, format='pdf', \
                bbox_inches='tight', \
                transparent=True)


def plot_energy_levels(Embl, Eeth, Ns, fname, W, Nlevels=None):
    """
    Plots the energy levels for the MBL and ETH states side-by-side
    
    Parameters:
    ----------

    Embl, Eeth:     arrays of energy eigenvalues for the MBL and ETH phases
    Ns:             number of sites
    fname:          output figure name
    W:              disorder strength as fraction of J
    Nlevels:        number of levels to plot (None -- plot all levels)

    """

    # Get the number of levels to plot
    Nl = Nlevels if Nlevels is not None else Embl.size

    # Setup of the figure
    # Default font sizes
    fsize = 20
    
    # Create a separate figure for each key
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    
    # Set the axes fontsizes
    set_axes_fonts(ax, fsize)

    # Set the locations of the energy levels
    n1range = np.linspace(0, 0.25, 100)
    n2range = np.linspace(0.75, 1, 100)
    erange = np.ones(n1range.size)

    # Iterate over all levels
    for Em, Ee in zip(Embl, Eeth):

        # Plot the energies as different colors
        ax.plot(n1range, Em*erange, 'k-')
        ax.plot(n2range, Ee*erange, 'r-')

    # Set the axes labels
    ax.set_ylabel(r'Energies [MHz]', fontsize=fsize)

    # Set title
    ax.set_title(r'Energy Levels (%d Sites)' % Ns, fontsize=fsize)

    # Set the ylims
    ax.set_ylim([-410, 410])

    # Add Annotation for the gap ratio
    gap_ratio = np.abs((Embl[1] - Embl[0]) / (Eeth[1] - Eeth[0]))
    ax.annotate('Disorder\n    %dJ' % W, xy=(0.4, 0.25), \
            xycoords='axes fraction', fontsize=fsize)
    ax.annotate('Gap Ratio\n    %.2f' % gap_ratio, xy=(0.4, 0.05), \
            xycoords='axes fraction', fontsize=fsize)

    # Save the results to file
    fig.savefig(fname, format='pdf', transparent=True)


