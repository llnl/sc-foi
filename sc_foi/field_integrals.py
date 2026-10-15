"""
Compute the overlap integrals Akk'(LJ, LJ0)
"""

import itertools
import re as regex
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import scipy.optimize

import prof_tools as pt


class EMFieldIntegrals:
    """
    Class to handle the calculation of overlap integrals
    """

    def __init__(self, LJs, LJstrs, modes, data_path,
                 *args, debug=False, **kwargs):
        """
        Constructor -- loads data from file
        """
        # Default settings for filenames
        self.dstr                  = '221112'
        self.has_updated_fields    = False
        self.has_normalized_fields = False
        self.normalize_fields      = not self.has_normalized_fields
        self.tt                    = pt.TimeTracker(print_times=debug)
        self.isvector              = True
        self.drop_max_field        = 0
        self.seed                  = 1889
        
        # Element type dictionary
        self.etype_dict = {'point' : 0, 'line' : 1, 'triangle' : 2, 
                           'quad' : 3, 'tetrahedral' : 4, 'pyramid' : 5,
                           'wedge' : 6, 'hexahedron' : 7}
        
        self.etype = 'tetrahedral'
        
        # Mesh length units in meters [default is mm]
        self.mesh_scale = 1e-3

        # Default plot settings
        self.fsize = 20
        self.lsize = 14

        # XXX: This is the old aedplt format, now working with csv format
        self.valid_groups = {'BoundingBox' : float, 'Elements' : int,
                             'Nodes'       : float, 'ElemSolution' : float,
                             'ElemSolutionMinMaxLocation' : float}

        # Standard class initialization as members from arguments and keyword
        # arguments, i.e. absorb all inputs into the class
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Update the string data after reading the keyword arguments
        ustr                 = '_updated' if self.has_updated_fields else ''
        nstr                 = '_norm' if self.has_normalized_fields else ''
        # self.filename_format       = f'efield_mode%d_LJ%snH_%s{ustr}{nstr}.fld'
        self.filename_format = None # f'efields_mode_%d_LJ%snH{nstr}.aedtplt'

        # Standard class initialization as members from arguments and keyword
        # arguments, i.e. absorb all inputs into the class
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # For statistical and optimization functions
        self.rng = np.random.default_rng(self.seed)
        
        # Default state of fields stored in class
        self.has_fields = False

    def read_group_from_aedtplt_file(self, filename, group_string,
                                     group_type=float):
        """
        Reads contents of a group from an aedtplt file from HFSS
        """
        # Check that the group_string is in the valid_groups
        if not (group_string in self.valid_groups.keys()):
            raise KeyError(f'Invalid group_string ({group_string}).')

        # Read the strdata from file
        with open(filename, 'r') as fid:
            strdata = fid.read()

        # Split the data by comma and convert to group_type
        strlist = regex.findall(f'{group_string}(.*)', strdata)[0].split(',')
        grplist = [group_type(s.replace('(', '').replace(')', '').strip()) \
                for s in strlist]

        return np.asarray(grplist)

    def group_to_arrays(self, group, group_name, return_3d_nodes=True):
        """
        Converts a group into the appropriate arrays
        """
        if group_name == 'Nodes':
            # Check that the nodes are divisible by 3
            mod3 = (len(group) % 3)
            assert not mod3, f'Nodes not divisible by 3 with remainder {mod3}.'
            group = group.reshape([len(group)//3, 3])
            x = group[:, 0]
            y = group[:, 1]
            z = group[:, 2]

            # Return as x,y,z or composite array
            if return_3d_nodes:
                return x, y, z
            else:
                return group

        elif group_name == 'Elements':
            # Read the total number of nodes and elements
            Ntotnodes = group[0]
            Ntotelements = group[1]

            # Group the elements by their node indices
            elems = []
            elements = group[2:]
            offset = 0
            for i in range(Ntotelements):
                el = elements[i*offset:i*offset+5]
                etype, id1, id2, id3, N = el
                # Indices are 1-indexed, shift to 0-indexed
                idxs = elements[i*offset+5:i*offset+(5+N)] - 1
                elems.append(idxs)
                offset = (5 + N)
            
            # Set the element type
            keys = list(self.etype_dict.keys()) 
            vals = list(self.etype_dict.values())
            self.etype = keys[vals.index(etype)]

            elems = np.asarray(elems)

            return elems

        elif group_name == 'ElemSolution':
            # Read off the minimum/maximum field values
            field_min, field_max, Nfields = group[0:3]
            field_data = group[3:]
            Ndim = 3 if self.isvector else 1
            Nnodes = int(Nfields) // Ndim
            Nelem = field_data.size // (Ndim * Nnodes)

            # print(f'Ndim: {Ndim}')
            # print(f'Nnodes: {Nnodes}')
            # print(f'field_data.shape: {field_data.shape}')
            # print(f'field_data.size / ({Nnodes} x {Ndim}): {Nelem}')

            # Group the elements by their node indices
            fields = []
            # print(f'Nelem: {Nelem}')
            for i in range(Nelem):
                idx0 = i*Ndim*Nnodes
                idx1 = (i+1)*Ndim*Nnodes
                fld = field_data[idx0:idx1]
                # if (i % Nelem) == 0:
                    # print(f'idx[{i}]: [{idx0}:{idx1}]')
                    # print(f'fld: {fld}')
                fld = fld.reshape([Nnodes, Ndim])
                # if (i % Nelem) == 0:
                    # print(f'fld: {fld}')
                fields.append(fld)

            fields = np.asarray(fields)
            
            # if self.drop_max_field:
            #     # Compute the magnitudes of fields
            #     for i in range(self.drop_max_field):
            #         fmag = np.sqrt(fields[:, :, 0]**2
            #                      + fields[:, :, 1]**2 
            #                      + fields[:, :, 2]**2)
            #         fmax = np.max(fmag); fmed = np.median(fmag)
            #         # print(f'Compared to median field with {fmed} V/m amplitude ...')
            #         
            #         # Find the indices of the maximum field and drop all nodes
            #         # in the corresponding element
            #         fmaxidx = np.unravel_index(np.argmax(fmag), fmag.shape)
            #         
            #         # Zero out one field
            #         # fields[fmaxidx[0], fmaxidx[1], :] = np.zeros(3)
            #         
            #         # Zero out entire element
            #         # fields[fmaxidx[0], :, :] = np.zeros([Nnodes, 3])
            #         
            #         # Delete entire element
            #         fields = np.delete(fields, fmaxidx, axis=0)
            #         
            #         # # Interpolate one field from intra-element values
            #         # ## Try the parallel vector trick
            #         # pvec = self.find_parallel_vector(fields[fmaxidx[0], :, :])
            #         # fields[fmaxidx[0], fmaxidx[1], :] = pvec
            #     
            #     return fmaxidx, fields
              
            # print(f'fields.shape: {fields.shape}')

            return fields

        else:
            raise ValueError(f'group_name {group_name} not recognized.')

    def nodes_at_elements(self, elements, nodes):
        """
        Returns the node values at each element node index
        """
        # Get the dimensions of the elements
        Nelem, Nnodes = elements.shape

        # Iterate over each element and get the nodes
        # print(f'nodes.shape: {nodes.shape}')
        # print(f'elements.shape: {elements.shape}')
        ordered_nodes = np.zeros([Nelem, Nnodes, 3])
        for i in range(Nelem):
            ordered_nodes[i,:] = nodes[elements[i,:]]

        return ordered_nodes
    
    def fields_at_elements(self, elements, fields):
        """
        Returns the node values at each element node index
        """
        # Get the dimensions of the elements
        Nelem, Nnodes = elements.shape

        # Iterate over each element and get the nodes
        ordered_fields = np.zeros([Nelem, Nnodes, 3])
        # print(f'fields.shape: {fields.shape}')
        # print(f'elements.shape: {elements.shape}')
        for i in range(Nelem):
            ordered_fields[i,:] = fields[elements[i,:]]

        return ordered_fields

    def load_fields(self, return_fields=False, return_vols=False,
                    write_norm_file=False):
        """
        Read data from file into a structure of mode numbers and LJs
        """
        # Start the fields as empty lists, likely ragged arrays
        fields             = {}
        abscissa           = {}
        vols               = {}
        Vtots              = {}
        LJs                = self.LJs
        modes              = self.modes
        LJstrs             = self.LJstrs
        debug              = self.debug
        data_path          = self.data_path

        # Filename formatting and date string (dstr)
        filename_format = self.filename_format
        dstr            = self.dstr

        # Iterate over modes and LJ
        self.tt.set_timer('load_fields()')
        print(f'load_fields() modes: {modes}')
        for kidx, k in enumerate(modes):
            print(f'Loading Mode-{k} ...')
            for LJidx, LJ in enumerate(LJs):
                # Construct the filenames on the fly
                if data_path != '':
                    fname = f'{data_path}/' + \
                    filename_format % (k, LJstrs[LJidx]) # , dstr)
                else:
                    fname = filename_format % (k, LJstrs[LJidx]) # , dstr)

                abss = self.read_group_from_aedtplt_file(fname,
                            'Nodes', group_type=float)
                nodes = self.group_to_arrays(abss, 'Nodes',
                            return_3d_nodes=False)
                
                elements_grp = self.read_group_from_aedtplt_file(fname,
                            'Elements', group_type=int)
                elements = self.group_to_arrays(elements_grp, 'Elements')

                flds_grp = self.read_group_from_aedtplt_file(fname,
                            'ElemSolution', group_type=float)
                
                # Get the ordered node and field data
                ordered_nodes = self.nodes_at_elements(elements, nodes)
                
                # if self.drop_max_field:
                #     fmaxidx, flds = self.group_to_arrays(flds_grp, 'ElemSolution')
                #     ordered_nodes = np.delete(ordered_nodes, fmaxidx, axis=0)
                # else:
                #     flds = self.group_to_arrays(flds_grp, 'ElemSolution')
                flds = self.group_to_arrays(flds_grp, 'ElemSolution')

                # Reshape to suppress the element information
                Nf1, Nf2, Nf3 = flds.shape
                
                # Compute the tetrahedral element volumes
                Nelems, Nnodes, Ndim = ordered_nodes.shape
                vol = np.zeros(Nelems)
                Vtot = 0
                debug = 0
                for n in range(Nelems):
                    # vol[n] = self.get_tetrahedron_volume(ordered_nodes[n, :, :],
                    #                                      debug=debug)
                    vol[n] = self.get_element_volume(ordered_nodes[n, :, :],
                                                         debug=debug)
                    Vtot += vol[n]

                # Construct the keys for the dictionaries
                key = f'k_{k}_LJ_{LJstrs[LJidx]}'
                if debug:
                    print(f'Building fields for {key} ...')

                # Normalize the fields
                if self.normalize_fields:
                    flds_norm = self.normalize_fields_points(flds, vol, Vtot)
                else:
                    flds_norm = np.copy(flds)

                # Store the abscissa and fields
                abscissa.update({key : ordered_nodes})
                fields.update({key : flds_norm})
                vols.update({key : vol})
                Vtots.update({key : Vtot})

                # Write the normalized fields to file
                if write_norm_file:
                    fsplit = fname.split('.')
                    fext = fsplit[-1]
                    filename_out = ''.join(fsplit[0:-1]) \
                                    + f'_norm.{fext}'
                    self.tt.set_timer(f'write_file({filename_out})')
                    with open(filename_out, 'w') as fid:
                        fid.write('\n'.join([\
                        f'{x1}, {y1}, {z1}, {Exr}, {Eyr}, {Ezr}'\
                         for x1, y1, z1, Exr, Eyr, Ezr in \
                         zip(xx, yy, zz, Ex, Ey, Ez) ]))
                    self.tt.get_timer()
        self.tt.get_timer()

        # Store fields and abscissa in the class
        self.fields     = fields
        self.abscissa   = abscissa
        self.has_fields = True
        self.vols       = vols
        self.Vtots      = Vtots

        # Return data as needed
        if return_fields and not return_vols:
            return abscissa, fields
        elif return_fields and return_vols:
            return abscissa, fields, vols, Vtots
            
    def plot_abscissa(self, xyz, fname):
        """
        Plot the positions of a given field plot
        """
        # Get the matplotlib wrapper object
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for x in xyz:
            ax.scatter(x[0], x[1], x[2], marker='.', s=5)
            ax.set_box_aspect((np.ptp(x[0]), np.ptp(x[1]), np.ptp(x[2])))
        ax.set_xlabel('x', fontsize=self.fsize)
        ax.set_ylabel('y', fontsize=self.fsize)
        ax.set_zlabel('z', fontsize=self.fsize)

        # Write the results to file
        fext = fname.split('.')[-1]
        fig.savefig(fname, format=fext, transparent=True)
        
    def plot_tetrahedron(self, xyz, fname):
        """
        Plots a single tetrahedron
        """
        xyz = np.asarray(xyz)
        # Get the matplotlib wrapper object
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        idx = 0
        # print(f'xyz.shape: {np.shape(xyz)}')
        for x in xyz.T:
            # print(f'x: {x}')
            ax.scatter(x[0], x[1], x[2], marker='o', s=30, color='b')
            ax.text(x[0], x[1], x[2], f'{idx+1}', size=20, zorder=1, color='k')
            idx += 1
        ax.set_box_aspect((np.ptp(xyz[0, :]), np.ptp(xyz[1, :]), np.ptp(xyz[2, :])))
        ax.set_xlabel('x', fontsize=self.fsize)
        ax.set_ylabel('y', fontsize=self.fsize)
        ax.set_zlabel('z', fontsize=self.fsize)

        # Write the results to file
        fext = fname.split('.')[-1]
        fig.savefig(fname, format=fext, transparent=True)
        
    def get_element_volume(self, element, debug=False):
        """
        Computes the volume, or area, of a general element
        """
        if self.etype == 'tetrahedral':
            return self.get_tetrahedron_volume(element, debug=debug)
        elif self.etype == 'triangle':
            return self.get_triangle_area(element, debug=debug)
        else:
            raise KeyError(f'Element type ({self.etype}) not supported.')
        
    def get_triangle_area(self, element, debug=False):
        """
        Computes the area of a triangular element
        """
        # If linear element
        n1 = element[0, 0:2].reshape([1, 2])
        n2 = element[1, 0:2].reshape([1, 2])
        n3 = element[2, 0:2].reshape([1, 2])
        
        mat = np.vstack((n1, n2, n3))
        mat = np.hstack((mat, np.ones((3, 1))))
        
        area = 0.5 * np.abs(np.linalg.det(mat))
        
        # Check that the vertices do not all lie along a line
        if area > 0:
            return area
        else:
            # If parabolic element
            n3 = element[2, 0:2].reshape([1, 2])
            mat = np.vstack((n1, n2, n3))
            mat = np.hstack((mat, np.ones((3, 1))))
            
            area = 0.5 * np.abs(np.linalg.det(mat))
            
            return area
        
    def get_tetrahedron_volume(self, element, debug=False):
        """
        Computes the volume of a tetrahedral element
        """
        # Get the node vertices, assuming they are not the first 4
        # They are Lagrange parabolic:  1, 3, 6, 10
        n1 = element[0, :].reshape([3, 1])
        n2 = element[2, :].reshape([3, 1])
        n3 = element[5, :].reshape([3, 1])
        n4 = element[9, :].reshape([3, 1])
        
        mat = np.hstack((n1, n2, n3, n4))
        mat = np.vstack((mat, np.ones(4)))
        mat_copy = np.copy(mat)
        
        # Cover all permutation of n1, n2, n3, n4 until vol > 0
        for idx in itertools.permutations([0, 1, 2, 3]):
            vol = self.mesh_scale**3 * np.linalg.det(mat_copy[:, idx]) / 6
            if vol > 0:
                break
            
        return vol
    
    def find_parallel_vector(self, u):
        """
        Computes a vector v that maximizes its dot product with sum_j u_j
        where u = {u_0, u_1, ..., }
        """
        u = np.asarray(u)
        N1, N2 = u.shape
        x0 = self.rng.standard_normal(3)
        def vdotu(x):
            return -sum([x[0] * u[i, 0]
               + x[1] * u[i, 1]
               + x[2] * u[i, 2]
                 for i in range(N1)])
        
        res = scipy.optimize.minimize(vdotu, x0=x0)
        
        return res.x

    def plot_fields(self, xyz, ExEyEz, fname,
                    # cmap_str : str = 'viridis',
                    cmap_str : str = 'jet',
                    plot_scale : str = 'linear',
                    plot_vector : bool = True,
                    transparent : bool = True,
                    plot_single_tet : list = None):
        """
        Plot the fields at their respective positions
        """
        # Get the positions and field components
        Nx, Ny, Nz     = np.shape(xyz)
        NEx, NEy, NEz  = np.shape(ExEyEz)
        print(f'ExEyEz.shape: {np.shape(ExEyEz)}')
        
        # Get the matplotlib wrapper object
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.grid(False)
        # ax.set_axis_off()
        if plot_scale == 'log':
            # Enorm = np.log10(np.sqrt(Ex**2 + Ey**2 + Ez**2))
            Enorm = np.log10(np.sqrt(ExEyEz[:,:,0].reshape([NEx * NEy])**2 
                                   + ExEyEz[:,:,1].reshape([NEx * NEy])**2 
                                   + ExEyEz[:,:,2].reshape([NEx * NEy])**2))
        elif plot_scale == 'linear':
            # Enorm = np.sqrt(Ex**2 + Ey**2 + Ez**2)
            Enorm = np.sqrt(ExEyEz[:,:,0].reshape([NEx * NEy])**2 
                          + ExEyEz[:,:,1].reshape([NEx * NEy])**2 
                          + ExEyEz[:,:,2].reshape([NEx * NEy])**2)
        else:
            raise KeyError(f'({plot_scale}) not recognized plot scale.')

        # Color map for the vector field / intensity plot
        alpha = 0.5 if transparent else 1.0
        cmap = mpl.cm.get_cmap(cmap_str)
        colors = cmap(Enorm, alpha=alpha)
        
        if plot_single_tet is not None:
            idxs = plot_single_tet
        else:
            idxs = list(range(Nx))

        # Vector field plot
        if plot_vector:
            for i in idxs:
                for j in range(Ny):
                    Enorm = np.sqrt(ExEyEz[i,j,0]**2 
                                  + ExEyEz[i,j,1]**2 
                                  + ExEyEz[i,j,2]**2)
                    colors = cmap(Enorm, alpha=alpha)
                    ax.quiver(xyz[i, j, 0], 
                              xyz[i, j, 1], 
                              xyz[i, j, 2],
                              ExEyEz[i, j, 0], 
                              ExEyEz[i, j, 1], 
                              ExEyEz[i, j, 2],
                              color=colors,
                              length=0.1,
                              normalize=True)
        else:
            for i in idxs:
                for j in range(Ny):
                    Enorm = np.sqrt(ExEyEz[i,j,0]**2 
                                  + ExEyEz[i,j,1]**2 
                                  + ExEyEz[i,j,2]**2)
                    colors = cmap(Enorm, alpha=alpha)
                    # ax.scatter(xyz[i, j, 0], 
                    #           xyz[i, j, 1], 
                    #           xyz[i, j, 2], color='tab:blue') # ,
                              # ExEyEz[i, j, 0], 
                              # ExEyEz[i, j, 1], 
                              # ExEyEz[i, j, 2])

        # ax.set_box_aspect((np.ptp(xyz[:,:,0]), np.ptp(xyz[:,:,1]), np.ptp(xyz[:,:,2])))

        ax.set_box_aspect((np.ptp(ExEyEz[:, :, 0].reshape([Nx * Ny])), 
                           np.ptp(ExEyEz[:, :, 1].reshape([Nx * Ny])),
                           np.ptp(ExEyEz[:, :, 2].reshape([Nx * Ny]))))

        # # Geometry axis labeling
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        # Write the results to file
        if fname != '':
            fext = fname.split('.')[-1]
            print(f'Writing {fname} ...')
            fig.savefig(fname, format=fext, transparent=True)
        
    def plot_field_histogram(self, xyz, ExEyEz, fname, cmap_str='viridis',
                    plot_scale='linear', plot_vector=False,
                    transparent=True, plot_single_tet=0, bins=70):
        """
        Plot the fields at their respective positions
        """
        # Get the positions and field components
        Nx, Ny, Nz     = np.shape(xyz)
        NEx, NEy, NEz  = np.shape(ExEyEz)
        
        Nxx = plot_single_tet if plot_single_tet > 0 else Nx

        # Get the matplotlib wrapper object
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        # Vector field plot
        if plot_vector:
            pass
        else:
            for i in range(Nxx):
                for j in range(Ny):
                    Enorm = np.sqrt(ExEyEz[i,j,0]**2 
                                  + ExEyEz[i,j,1]**2 
                                  + ExEyEz[i,j,2]**2)
                    ax.hist(Enorm, color='b')
                    
        ax.set_xlabel(r'$|E|$ [V/m]', fontsize=20)
        ax.set_ylabel(r'Counts', fontsize=20)

        # Write the results to file
        fext = fname.split('.')[-1]
        print(f'Writing {fname} ...')
        fig.savefig(fname, format=fext, transparent=True)

    def strip_nans(self, filename, nan_str='Nan', update_file=True,
                   header_skip=2):
        """
        Removes the nan string and associated row from the file
        """
        # Read the strdata from file
        with open(filename, 'r') as fid:
            strdata = fid.read()

        # Drop the first header_skip rows
        if not self.has_updated_fields:
            strdata = '\n'.join(strdata.split('\n')[header_skip:])
        else:
            strdata = np.loadtxt(filename, dtype=float)

            return strdata

        # Find all occurences of NAN and remove
        # print(f'Removing all rows with \'{nan_str}\' string ...')
        no_nan_rows = regex.sub(f'(.*){nan_str} \n', '', strdata).split('\n')
        no_nan_rows = [n for n in no_nan_rows if n != '']

        strd = [s.split(' ') for s in no_nan_rows]
        strd = np.asarray([[float(ss.strip()) for ss in s if ss != ''] \
                            for s in strd])

        # Write the new data to file
        if update_file:
            fsplit = filename.split('.')
            fext = fsplit[-1]
            filename_out = ''.join(fsplit[0:-1]) + f'_updated.{fext}'
            # print(f'Updating file at {filename_out} ...')
            with open(filename_out, 'w') as fid:
                fid.write('\n'.join(no_nan_rows))

        return strd

    def build_efields(self, return_fields=False, write_norm_file=False):
        """
        Read data from file into a structure of mode numbers and LJs
        """
        # Start the fields as empty lists, likely ragged arrays
        fields             = {}
        abscissa           = {}
        LJs                = self.LJs
        modes              = self.modes
        LJstrs             = self.LJstrs
        debug              = self.debug
        data_path          = self.data_path

        # Filename formatting and date string (dstr)
        filename_format = self.filename_format
        dstr            = self.dstr
        
        # Iterate over modes and LJ
        self.tt.set_timer('strip_nans()')
        for kidx, k in enumerate(modes):
            for LJidx, LJ in enumerate(LJs):
                # Construct the filenames on the fly
                fname = f'{data_path}/' + \
                        filename_format % (k, LJstrs[LJidx], dstr)
                edata = self.strip_nans(fname)

                # Construct the keys for the dictionaries
                key = f'k_{k}_LJ_{LJstrs[LJidx]}'
                if debug:
                    print(f'Building fields for {key} ...')

                # Read off the fields and abscissa
                ## (x, y, z)
                ## (Ex, Ey, Ez)
                x = edata[:, 0:3]
                E = edata[:, 3:]
            
                # Normalize the fields
                if self.normalize_fields:
                    E = self.normalize(x, E)

                # Store the abscissa and fields
                abscissa.update({key : x})
                fields.update({key : E})

                # Write the normalized fields to file
                if write_norm_file:
                    fsplit = fname.split('.')
                    fext = fsplit[-1]
                    filename_out = ''.join(fsplit[0:-1]) \
                                    + f'_norm.{fext}'
                    self.tt.set_timer(f'write_file({filename_out})')
                    with open(filename_out, 'w') as fid:
                        fid.write('\n'.join([\
                        f'{x} {y} {z} {Exr} {Exi} {Eyr} {Eyi} {Ezr} {Ezi}'\
                         for x, y, z, Exr, Exi, Eyr, Eyi, Ezr, Ezi in \
                         zip(x[:, 0], x[:, 1], x[:, 2],
                             E[:, 0], E[:, 1], E[:, 2],
                             E[:, 3], E[:, 4], E[:, 5]) ]))
                    self.tt.get_timer()
        self.tt.get_timer()

        # Store fields and abscissa in the class
        self.fields     = fields
        self.abscissa   = abscissa
        self.has_fields = True

        # Return data as needed
        if return_fields:
            return abscissa, fields

    def normalize(self, x, E):
        """
        Computes the normalization of a field by dividing by its energy, the
        square root of the integral of its amplitude squared
        """
        # Compute the integrands (inner products) of complex vector E
        dot_prods = np.asarray([EE[0::2].dot(EE[1::2]) for EE in E])

        # Interpolate the dot products and pass to the integrator
        interp = scipy.interpolate.NearestNDInterpolator(x, dot_prods)
        def finterp(xx, yy, zz):
            return interp(xx, yy, zz)

        # Compute the volume integral using the coordinate ranges
        dx1 = [x[:,0].min(), x[:,0].max()]
        dx2 = [x[:,1].min(), x[:,1].max()]
        dx3 = [x[:,2].min(), x[:,2].max()]

        print('Computing field normalization ...')
        opts = {'limit' : 1000}
        norm, err = scipy.integrate.nquad(finterp, [dx1, dx2, dx3], opts=opts)
        norm = abs(norm)

        if self.debug:
            print(f'normalize(): {norm:.2g} quadrature error: {err:.2g}')

        return E / norm
    
    def normalize_fields_points(self, f, vols, Vtot):
        """
        Normalizes the fields to the energy integral, L2-norm
        """
        # Compute the overlaps as the dot products
        fdims  = f.shape
        Nelems = fdims[0]
        Nnodes = fdims[1]
        Imn = 0
        for m in range(Nelems):
            for n in range(Nnodes):
                # Accumulate the dot products
                Imn += (f[m, n, 0]**2
                      + f[m, n, 1]**2
                      + f[m, n, 2]**2) * vols[m]
                
            # Scale by the tetrahedral volumes
            # Imn *= vols[m]
        
        # Normalize by the total volume
        Imn /= Vtot
        fnormed = f / np.sqrt(Imn)
        
        return fnormed
    
    def overlap_integral_points(self, f, fp, vols, volsp, Vtot, Vtotp):
        """
        Computes and returns the overlap integral for a particular k/k', LJ/LJ'
        quadruple using the points as the finite grid to compute the integrals
        """
        # Get the field dimensions
        fdims  = f.shape
        fpdims = fp.shape
        Nelems = min(fdims[0], fpdims[0])
        Nnodes = min(fdims[1], fpdims[1])
        
        # Drop the maximum field values
        if self.drop_max_field:
            Nnodes = f.shape[1]
            # Compute the magnitudes of fields
            for i in range(self.drop_max_field):
                fmag = np.sqrt(f[:, :, 0]**2
                             + f[:, :, 1]**2 
                             + f[:, :, 2]**2)
                fmax = np.max(fmag); fmed = np.median(fmag)
                fpmag = np.sqrt(fp[:, :, 0]**2
                              + fp[:, :, 1]**2 
                              + fp[:, :, 2]**2)
                fmax = np.max(fmag); fmed = np.median(fmag)
                fpmax = np.max(fpmag); fpmed = np.median(fpmag)
                
                # Find the indices of the maximum field and drop all nodes
                # in the corresponding element
                fmaxidx = np.unravel_index(np.argmax(fmag), fmag.shape)
                fpmaxidx = np.unravel_index(np.argmax(fpmag), fpmag.shape)
                
                # # Delete entire element
                # f = np.delete(f, fmaxidx, axis=0)
                # # f[fmaxidx[0], :, :] = np.zeros([Nnodes, 3]) # = np.delete(f, fmaxidx, axis=0)
                # Vtot -= vols[fmaxidx[0]]
                # vols = np.delete(vols, fmaxidx[0])
                # fp = np.delete(fp, fpmaxidx, axis=0)
                # # fp[fpmaxidx[0], :, :] =  np.zeros([Nnodes, 3]) # np.delete(fp, fpmaxidx, axis=0)
                # Vtotp -= vols[fpmaxidx[0]]
                # volsp = np.delete(volsp, fpmaxidx[0])
            
        # Take the geometric average of the volumes
        Vtot_avg = np.sqrt(Vtot * Vtotp)
        
        # Compute the overlaps as the dot products
        Imn = 0
        fdims  = f.shape
        fpdims = fp.shape
        Nelems = min(fdims[0], fpdims[0])
        Nnodes = min(fdims[1], fpdims[1])
        N = Nelems * Nnodes
        for m in range(Nelems):
            dV = np.sqrt(vols[m] * volsp[m])
            for n in range(Nnodes):
                if self.drop_max_field:
                    c0 = (m == fmaxidx[0]) and (n == fmaxidx[1])
                    c1 = (m == fpmaxidx[0]) and (n == fpmaxidx[1])
                    if c0 or c1:
                        print(f'Dropping field at [{n}, {m}] ...')
                        continue
                # Accumulate the dot products
                Imn += (f[m, n, 0] * fp[m, n, 0] \
                      + f[m, n, 1] * fp[m, n, 1] \
                      + f[m, n, 2] * fp[m, n, 2]) * dV
                
        # Normalize by the total volume
        if self.normalize_fields:
            Imn /= Vtot_avg
        
        return Imn # / Vtot_avg

    def overlap_integral(self, x, xp, f, fp, Vtot=1.):
        """
        Computes and returns the overlap integral for a particular k/k' LJ/LJ'
        quadruple
        """
        # Complex inner product 
        def my_complex_inner(a, b):
            out = a[0] * b[0] + a[1] * b[1] \
                 - 1j*a[1] * b[0] + 1j*a[0] * b[1] \
                 + a[2] * b[2] + a[3] * b[3] \
                 - 1j*a[3] * b[2] + 1j*a[2] * b[3] \
                 + a[4] * b[4] + a[5] * b[5] \
                 - 1j*a[5] * b[4] + 1j*a[4] * b[5]
            return out
        
        # Real-valued inner product 
        def my_real_inner(a, b):
            out = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            return out

        # Take the real parts of the inner products
        dot_prods = np.asarray([my_real_inner(ff, ffp) \
                                for ff, ffp in zip(f, fp)])

        # Use the shortest of the two vector field abscissa
        x = np.copy(xp) if xp.size < x.size else np.copy(x)

        # Interpolate the dot products
        interp = scipy.interpolate.NearestNDInterpolator(x, dot_prods)
        
        def finterp(xx, yy, zz):
            return interp(xx, yy, zz)

        # Compute the volume integral using the coordinate ranges
#         dx1 = [x[:,0].min(), x[:,0].max()]
#         dx2 = [x[:,1].min(), x[:,1].max()]
#         dx3 = [x[:,2].min(), x[:,2].max()]

        dx1 = [x[:,0].min(), x[:,0].max()]
        dx2 = [x[:,1].min(), x[:,1].max()]
        dx3 = [x[:,2].min(), x[:,2].max()]
        
        # Integration by Chebyshev interpolation
        # Uses discrete cosine transform to compute coefficients
        # With the same scaling as FFT, O(N log N)
        # Not sure if the alpha and beta are optimal, but they seem to be
        # reasonably good with a 6x speedup and error reduction
        # The name of this quadrature rule is Clenshaw-Curtis:
        # https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature
        alpha = 0.5; beta = 0.5;
        opts = {'limit' : 50, 'epsrel' : 1e-6 , 'weight' : 'alg',
                'wvar' : (alpha, beta)}
        # print(f'interp?: {interp}')
        print(f'interp(dx1, dx2, dx3): {interp(dx1[0], dx2[0], dx3[0])}')
        val, err, neval = scipy.integrate.nquad(finterp, [dx1, dx2, dx3],
                                            opts=[opts]*3, full_output=True)
        nevals = neval['neval']

        if self.debug:
            print(f'overlap_integral(): {abs(val):.2g}, '\
                + f'quadrature error: {err:.2g}, neval: {nevals}')

        return abs(val) / Vtot

    def overlap_matrix(self, fname=None, return_matrix=False, Vtot=1.):
        """"
        Computes the A_{k,k'; LJ,LJ'} overlap integral matrix
        """
        # Start the fields as empty lists, likely ragged arrays
        Akk      = {}
        LJs      = self.LJs
        modes    = self.modes
        LJstrs   = self.LJstrs
        fields   = self.fields
        abscissa = self.abscissa

        # Iterate over modes and LJ, twice to build up off-diagonals
        for kidx, k in enumerate(modes):
            for kpidx, kp in enumerate(modes):
                for LJidx, LJ in enumerate(LJs):
                    for LJpidx, LJp in enumerate(LJs):
                        # Start the timer
                        start = time.time()
                        # Construct the keys for the dictionaries
                        key = f'k_{k}_LJ_{LJstrs[LJidx]}'
                        keyp = f'k_{kp}_LJ_{LJstrs[LJpidx]}'
                        fullkey = f'kkp{k}_{kp}_LJLJp_{LJstrs[LJidx]}_{LJstrs[LJpidx]}'
                        self.tt.set_timer(f'overlap({fullkey})')
                        print(f'Running {fullkey} ...')

                        # Read off the fields and abscissa
                        x  = abscissa[key].T
                        xp = abscissa[keyp].T
                        f  = fields[key].T
                        fp = fields[keyp].T
                        
                        # Compute and store the overlap matrix elements
                        Akk[fullkey] = self.overlap_integral(x, xp, f, fp, Vtot=Vtot)
                        
                        self.tt.get_timer()

        # Write results to file
        if fname:
            df = pd.DataFrame(Akk.values(), columns=Akk.keys())
            df.to_csv(fname)

        if return_matrix:
            return Akk

    def overlap_matrix_points(self, fname=None, return_matrix=False, cidxs=None, drop_points=1):
        """"
        Computes the A_{k,k'; LJ,LJ'} overlap integral matrix
        """
        # Start the fields as empty lists, likely ragged arrays
        Akk = {}
        LJs = self.LJs
        modes = self.modes
        LJstrs = self.LJstrs
        fields = self.fields
        vols = self.vols
        Vtots = self.Vtots
        
        Akkmat = np.zeros([len(LJs), len(LJs), len(modes), len(modes)])

        # Iterate over modes and LJ, twice to build up off-diagonals
        self.tt.set_timer('overlap_matrix_points()')
        for kidx, k in enumerate(modes):
            for kpidx, kp in enumerate(modes):
                for LJidx, LJ in enumerate(LJs):
                    for LJpidx, LJp in enumerate(LJs):
                        # Construct the keys for the dictionaries
                        key = f'k_{k}_LJ_{LJstrs[LJidx]}'
                        keyp = f'k_{kp}_LJ_{LJstrs[LJpidx]}'
                        fullkey = f'kkp{k}_{kp}_LJLJp_{LJstrs[LJidx]}_{LJstrs[LJpidx]}'

                        # Read off the fields and abscissa
                        f     = fields[key]
                        fp    = fields[keyp]
                        vol   = vols[key]
                        volp  = vols[keyp]
                        Vtot  = Vtots[key]
                        Vtotp = Vtots[keyp]
                        
                        # Compute and store the overlap matrix elements
                        print(f'Running {fullkey} ...')
                        
                        if cidxs:
                            c0_cond = (kidx + 1 == cidxs[0]) and (kpidx + 1 == cidxs[1])
                            c1_cond = (kidx + 1 == cidxs[1]) and (kpidx + 1 == cidxs[0])
                            cidx_condition = c0_cond or c1_cond
                            if cidx_condition and (kidx != kpidx):
                                print(f'Correcting [{kidx+1}, {kpidx+1}] ...')
                                self.drop_max_field = drop_points
                        # Akk[fullkey] = self.overlap_integral_points(f, fp, vol, volp, Vtot, Vtotp)
                        # Akkmat[LJidx, LJpidx, kidx, kpidx] = Akk[fullkey]
                        Akkmat[LJidx, LJpidx, kidx, kpidx] = self.overlap_integral_points(
                            f, fp, vol, volp, Vtot, Vtotp)
                        
                        # Reset the drop_max_field condition
                        self.drop_max_field = 0
                        
        self.tt.get_timer()

        # Write results to file
        if fname:
            df = pd.DataFrame(Akk.values(), columns=Akk.keys())
            df.to_csv(fname)

        if return_matrix:
            # return Akk
            return Akkmat
        
import pyEPR
from pyEPR.core_distributed_analysis import CalcObject


class JunctionParticipations:
    """
    Handles the junction participation calculations as the second
    term in the orthogonality condition defining the Akk', Bkk'
    coefficients
    """
    def __init__(self, eprh : pyEPR.DistributedAnalysis, Nmodes :  int,
                 LJs : np.ndarray, CJs : np.ndarray,
                 data_path : str,
                 jjstr : str= 'jj_line',
                 vstr : str = 'LJ', 
                 vunits : str = 'nH',
                 jj_area : float = (50e-6)**2, 
                 **kwargs):
        # Get the distributed analysis object
        # self.eprh    = eprh
        # self.LJstr   = LJstr
        # self.jjstr   = jjstr
        # self.jj_area = jj_area
        # self.Nmodes  = Nmodes
        # Single point of entry inputs to class
        modes = np.linspace(1, Nmodes, Nmodes, dtype=int)
        print(f'JunctionParticipations modes: {modes}')
    
        # Call this after the defaults have all been set
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)
    
        # Sort the variations by LJ values
        self.sort_variations()
    
    def sort_variations(self):
        """
        Sorts the variations by descending inductances
        """
        # Read off the inductances and x-offsets from the variations
        variations = self.eprh.solutions.list_variations()
        Ljstrs = []
        Ljvals = []
        vstr = self.vstr
        vunits = self.vunits

        # Iterate over all variations
        for v in variations:
            # Read the value and the string using regular expressions
            val = float(regex.findall(f'{vstr}=\'(.*?){vunits}', v)[0])
            vs = regex.findall(f'{vstr}=\'(.*?{vunits})', v)[0].replace('.', 'p')
            print(f'val: {val}, vs: {vs}')

            # Append the results to the strings and values arrays
            Ljstrs.append(f'{vstr}_{vs}')
            Ljvals.append(val)
            
        # Sort the indices according to the values
        Lj_sort_idxs = np.argsort(Ljvals)
        Ljvals = np.unique(np.asarray(Ljvals)[Lj_sort_idxs])
        Ljstrs = np.asarray(Ljstrs)[Lj_sort_idxs]
    
        # Apply the sorting indices to the variations
        self.vars_sorted = np.asarray(self.eprh.variations)[Lj_sort_idxs]
        
    def compute_efield_energy(self, calc, lv):
        """
        Compute the energy stored in the E or H field
        """
        field = 'E'
        phase = 0.
        Ek = calc.getQty(field)
        Dk = Ek.times_eps()
        Eks = Ek.conj()
        Eks = Eks.dot(Dk)
        Eks = Eks.real()
        Eks = Eks.integrate_vol(name='AllObjects')
        Ekint = Eks.evaluate(lv=lv, phase=phase)

        return Ekint
    
    def compute_hfield_energy(self, calc, lv):
        """
        Compute the energy stored in the E or H field
        """
        field = 'H'
        phase = 90.
        Hk = calc.getQty(field)
        Bk = Hk.times_mu()
        Hks = Hk.conj()
        Hks = Hks.dot(Bk)
        Hks = Hks.real()
        Hks = Hks.integrate_vol(name='AllObjects')
        Hkint = Hks.evaluate(lv=lv, phase=phase)

        return Hkint

    def compute_pL_pC(self, return_matrix=True):
        """
        Compute the inductive and capacitive participations from
        the appropriate field integrals
        """
        # Initialize the participation "matrices"
        vlen = len(self.vars_sorted)
        # pLkkp = np.zeros([vlen, vlen, self.Nmodes, self.Nmodes])
        # pCkkp = np.zeros([vlen, vlen, self.Nmodes, self.Nmodes])
        pLkkp = np.zeros([vlen, self.Nmodes, self.Nmodes])
        pCkkp = np.zeros([vlen, self.Nmodes, self.Nmodes])

        # Get the inductances and capacitances
        LJs = self.LJs
        CJs = np.asarray(self.CJs).flatten()
        eprh = self.eprh

        # Iterate over all modes and variations
        for k in range(self.Nmodes):
            for kp in range(self.Nmodes):
                print(f'Modes ({k}, {kp}) ...')
                for idx, v in enumerate(self.vars_sorted):
                    for idx0, v0 in enumerate(self.vars_sorted):
                        # Get the LJ, LJ0, CJ, CJ0
                        LJ  = LJs[idx]
                        CJ = CJs[idx]
                        LJ0 = LJs[idx0]
                        CJ0 = CJs[idx0]
                        if (len(CJs) > idx) and (len(CJs) > idx0):
                            CJ = CJs[idx]
                            CJ0 = CJs[idx0]
    
                        # Set the solution to mode k
                        eprh.solutions.set_mode(k+1)
                        calc = CalcObject([], eprh.setup)
    
                        ## Compute the voltages
                        Vkre = calc.getQty('E').real().integrate_line_tangent(name=self.jjstr)
                        Vkim = calc.getQty('E').imag().integrate_line_tangent(name=self.jjstr)

                        # Compute the impedance Zk
                        # Set the variation to v
                        eprh.set_variation(v)
                        lv = eprh._get_lv(v)
                        wk, _ = eprh.solutions.eigenmodes(lv=lv)
                        wk = 2 * np.pi * np.real(np.asarray(wk))[k]
                        
                        # fk = CalcObject(
                        #      [('EnterOutputVar', ('Freq', "Complex"))], self.eprh.setup).real().evaluate()
                        # wk = 2*np.pi*fk  # in SI radian Hz units
                        # Z = jwL || 1/jwC
                        Zk = 1j * wk * LJ / (1 - wk**2 * LJ * CJ)
                        
                        # Vk = np.sqrt(Vkre.evaluate(lv=lv)**2 
                        #            + Vkim.evaluate(lv=lv)**2)
                        Vk = Vkre.evaluate(lv=lv) + 1j * Vkim.evaluate(lv=lv)
                        
                        ## Compute the electric, magnetic field integrals
                        Ekint = self.compute_efield_energy(calc, lv)
                        Hkint = self.compute_hfield_energy(calc, lv)
                        
                        # Set the solution to mode kp
                        eprh.solutions.set_mode(kp+1)
                        # Set the variation to v0
                        eprh.set_variation(v0)
                        lv0 = eprh._get_lv(v0)
                        calc = CalcObject([], eprh.setup)
                        Vkpre = calc.getQty('E').real().integrate_line_tangent(name=self.jjstr)
                        Vkpim = calc.getQty('E').imag().integrate_line_tangent(name=self.jjstr)
                        
                        # Compute the impedance Zk
                        # fkp = CalcObject(
                        #      [('EnterOutputVar', ('Freq', "Complex"))], eprh.setup).real().evaluate()
                        # wkp = 2*np.pi*fkp  # in SI radian Hz units
                        wkp, _ = eprh.solutions.eigenmodes(lv=lv0)
                        wkp = 2 * np.pi * np.asarray(wkp)[kp]
                        # Z = jwL || 1/jwC
                        Zkp = 1j * wkp * LJ0 / (1 - wkp**2 * LJ0 * CJ0)
                        # Zkp = wkp * LJ / (1 - wkp**2 * LJ * CJ)
                        
                        # Set the variation to v0
                        # eprh.set_variation(v0)
                        # lv0 = eprh._get_lv(v0)
                        #  Vkp = np.sqrt(Vkpre.evaluate(lv=lv0)**2 
                        #              + Vkpim.evaluate(lv=lv0)**2)
                        Vkp = Vkpre.evaluate(lv=lv0) + 1j * Vkpim.evaluate(lv=lv0)
                        # Vkp = np.sqrt(Vkpre.evaluate(lv=lv)**2 
                        #             + Vkpim.evaluate(lv=lv)**2)
                        
                        ## Compute the electric, magnetic field integrals
                        Ekpint = self.compute_efield_energy(calc, lv0)
                        Hkpint = self.compute_hfield_energy(calc, lv0)
                        # Ekpint = self.compute_efield_energy(calc, lv)
                        # Hkpint = self.compute_hfield_energy(calc, lv)
                        
                        # Compute the currents from the voltages and impedances
                        Ik = Vk / Zk
                        Ikp = Vkp / Zkp
                        
                        # Compute the numerators
                        numer_Lk = LJ * np.abs(Ik)**2
                        numer_Ck = CJ * Vk**2
                        # numer_Lkp = LJ * Ikp**2
                        # numer_Ckp = CJ * Vkp**2
                        numer_Lkp = LJ0 * np.abs(Ikp)**2
                        numer_Ckp = CJ0 * Vkp**2
                        
                        # Compute the denominators
                        denomEk = numer_Ck + Ekint
                        denomEkp = numer_Ckp + Ekpint
                        denomHk = numer_Lk + Hkint
                        denomHkp = numer_Lkp + Hkpint
                        
                        # Compute the participations
                        pLk = numer_Lk / denomHk
                        pLkp = numer_Lkp / denomHkp
                        
                        pCk = numer_Ck / denomEk
                        pCkp = numer_Ckp /  denomEkp
                        
                        # Return the correct ratios of participations
                        pLkkp[idx, k, kp] = 1. / np.sqrt(np.abs(pLk * pLkp)) 
                        pCkkp[idx, k, kp] = 1. / np.sqrt(np.abs(pCk * pCkp))

        # Assign the pL and pC to class members
        self.pL = pLkkp
        self.pC = pCkkp
                        
        if return_matrix:
            return pLkkp, pCkkp
        
    def compute_current_overlaps(self, return_matrix=True):
        """
        Compute the product of the currents Ik Ik'
        """
        Ikkp = np.zeros([len(self.vars_sorted), len(self.vars_sorted), self.Nmodes, self.Nmodes])
        for k in range(self.Nmodes):
            for kp in range(self.Nmodes):
                for idx, v in enumerate(self.vars_sorted):
                    for idx0, v0 in enumerate(self.vars_sorted):
                        # Set the solution to mode k
                        eprh.solutions.set_mode(k+1)
                        calc = CalcObject([], eprh.setup)
                        calc = calc.getQty('Jsurf').mag().integrate_surf(name=self.jj)
                        
                        # Set the variation to v
                        eprh.set_variation(v)
                        lv = eprh._get_lv(v)
                        Ik = calc.evaluate(lv=lv, phase=90.) / self.j_area
                        
                        # Set the solution to mode kp
                        eprh.solutions.set_mode(kp+1)
                        calc = CalcObject([], eprh.setup)
                        calc = calc.getQty('Jsurf').mag().integrate_surf(name=self.jj)
                        
                        # Set the variation to v0
                        eprh.set_variation(v0)
                        lv0 = eprh._get_lv(v0)
                        Ikp = calc.evaluate(lv=lv0, phase=90.) / self.jj_area
                    
                        print(f'(k, kp): ({k+1}, {kp+1}), (LJ, LJ0): ({v}, {v0})')
                        Ikkp[idx, idx0, k, kp] = Ik * Ikp  
        
        self.Ikkp = np.copy(Ikkp)
        
        if return_matrix:
            return Ikkp
        
class FieldIntegrals:
    """
    Class to handle the calculation of overlap integrals
    eprh : pyEPR.DistributedAnalysis, Nmodes :  int,
                 LJs : np.ndarray, CJs : np.ndarray,
                 data_path : str,
                 *args, jjstr : str='jj', LJstr : str ='LJ', 
                 jj_area : float = (50e-6)**2, 
                 **kwargs
    """
    
    def __init__(self, 
                 eprh : pyEPR.DistributedAnalysis,
                 Nmodes : int,
                 data_path : str,
                 LJs : np.ndarray,
                 CJs : np.ndarray,
                 LJstrs : list,
                 hfield_format : str,
                 efield_format : str,
                 *args,
                 jjstr : str='jj',
                 LJstr : str ='LJ', 
                 jj_area : float = (50e-6)**2, 
                 debug : bool = False, 
                 use_junction_participations : bool = False,
                 **kwargs):
        """
        Constructor -- loads data from file
        """
        # Default settings for filenames
        self.dstr                  = '221112'
        self.has_updated_fields    = False
        self.has_normalized_fields = False
        self.normalize_fields      = not self.has_normalized_fields
        self.tt                    = pt.TimeTracker(print_times=debug)
        self.isvector              = True
        self.drop_max_field        = 0
        self.seed                  = 1889
        self.eprh                  = None
        
        # Element type dictionary
        self.etype_dict = {'point' : 0, 'line' : 1, 'triangle' : 2, 
                           'quad' : 3, 'tetrahedral' : 4, 'pyramid' : 5,
                           'wedge' : 6, 'hexahedron' : 7}
        
        self.etype = 'tetrahedral'
        
        # Mesh length units in meters [default is mm]
        self.mesh_scale = 1e-3

        # Default plot settings
        self.fsize = 20
        self.lsize = 14

        # XXX: This is the old aedplt format, now working with csv format
        self.valid_groups = {'BoundingBox' : float, 'Elements' : int,
                             'Nodes'       : float, 'ElemSolution' : float,
                             'ElemSolutionMinMaxLocation' : float}

        # Standard class initialization as members from arguments and keyword
        # arguments, i.e. absorb all inputs into the class
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Update the string data after reading the keyword arguments
        ustr                 = '_updated' if self.has_updated_fields else ''
        nstr                 = '_norm' if self.has_normalized_fields else ''
            
        # For statistical and optimization functions
        self.rng = np.random.default_rng(self.seed)
        
        # Default state of fields stored in class
        # self.has_fields = False
        if self.normalize_fields:
            print('Integrals will use normalized fields.')
        else:
            print('Integrals will use non-normalized fields.')
        
        # Create internal instances of the two field integrals classes
        modes = np.linspace(1, Nmodes, Nmodes, dtype=int)
        self.em_field_int = EMFieldIntegrals(LJs, LJstrs, modes, data_path,
                 *args, debug=False, **kwargs)

        if self.use_junction_participations:
            self.junc_pratios = JunctionParticipations(eprh, Nmodes,
                 LJs, CJs, data_path, jjstr=jjstr, LJstr=LJstr, 
                 jj_area=jj_area)
    
    def repair_ab_matrices(self, Akkp, Bkkp):
        """
        Correct for symmetry violations in A and B
        """
        # Repair the A and B matrices
        Akkpcopy = Akkp.copy()
        Bkkpcopy = Bkkp.copy()
        LJlen = len(self.LJs)
        A34 = np.zeros([LJlen, LJlen])
        A43 = np.zeros([LJlen, LJlen])
        B34 = np.zeros([LJlen, LJlen])
        B43 = np.zeros([LJlen, LJlen])
        
        for k1 in range(self.Nmodes):
            for k2 in range(self.Nmodes):
                for idx in range(len(self.LJs)):
                    for idx0 in range(len(self.LJs)):
                        if k1 != k2:
                            # Repair the Akkp
                            A34[idx, idx0] = Akkpcopy[idx, idx0, k1, k2]
                            A43[idx, idx0] = Akkpcopy[idx, idx0, k2, k1]
                            # Akkp[idx, idx0, k1, k2] = (A34[idx, idx0] - A43[idx, idx0]) / 2
                            # Akkp[idx, idx0, k2, k1] = (A43[idx, idx0] - A34[idx, idx0]) / 2
                            Akkp[idx, idx0, k1, k2] = (A34[idx, idx0] + A43[idx, idx0]) / 2
                            Akkp[idx, idx0, k2, k1] = (A43[idx, idx0] + A34[idx, idx0]) / 2
                            
                            # Repair the Bkkp
                            B34[idx, idx0] = Bkkpcopy[idx, idx0, k1, k2]
                            B43[idx, idx0] = Bkkpcopy[idx, idx0, k2, k1]
                            # Bkkp[idx, idx0, k1, k2] = (B34[idx, idx0] - B43[idx, idx0]) / 2
                            # Bkkp[idx, idx0, k2, k1] = (B43[idx, idx0] - B34[idx, idx0]) / 2
                            Bkkp[idx, idx0, k1, k2] = (B34[idx, idx0] + B43[idx, idx0]) / 2
                            Bkkp[idx, idx0, k2, k1] = (B43[idx, idx0] + B34[idx, idx0]) / 2
                            
        return Akkp, Bkkp
        
    def check_orthogonality(self):
        """
        Compute the on and off diagonal elements of delta_{kk'}
        for a fixed (LJ0, LJ=LJ0) pair
        """
        # Compute the field integrals
        ## Compute the E-fields
        self.em_field_int.filename_format = self.efield_format
        self.em_field_int.load_fields()
        print('Computing Akkp ...')
        Akkp = self.em_field_int.overlap_matrix_points(fname=None, cidxs=None,
                                      drop_points=1, return_matrix=True) 
        
        ## Compute the H-fields
        print('Computing Bkkp ...')
        self.em_field_int.filename_format = self.hfield_format
        self.em_field_int.load_fields()
        Bkkp = self.em_field_int.overlap_matrix_points(fname=None, cidxs=None,
                                      drop_points=1, return_matrix=True) 
        
        ## Repair the A, B matrices
        print('Repairing Akkp, Bkkp ...')
        Akkp, Bkkp = self.repair_ab_matrices(Akkp, Bkkp)
        
        return Akkp, Bkkp
        
    def plot_loaded_fields_at_mode_and_param(self, mode_idx : int,
                                             param_str : str,
                                             field_type : str = 'E',
                                             plot_vector : bool = True,
                                            plot_single_tet : list = None,
                                            fname : str = ''):
        '''
        Plots the fields using the internal class functions to
        load the fields from file
        '''
        # Set the field format based on electric or magnetic fields
        if field_type == 'E':
            print(f'filename_format: {self.em_field_int.filename_format}')
            self.em_field_int.filename_format = self.efield_format
            print(f'filename_format: {self.em_field_int.filename_format}')
        elif field_type == 'H':
            self.em_field_int.filename_format = self.hfield_format
        else:
            raise ValueError(f'field_type ({field_type}) not supported.')
            
        # Get the field and positions
        abscissa, fields, vols, Vtots = self.em_field_int.load_fields(return_fields=True,
                                                              return_vols=True)
        
        # Start the fields as empty lists, likely ragged arrays
        LJs = self.em_field_int.LJs
        modes = self.em_field_int.modes
        LJstrs = self.em_field_int.LJstrs

        # Iterate over modes and LJ, twice to build up off-diagonals
        self.tt.set_timer('overlap_matrix_points()')
        
        # Construct the keys for the dictionaries
        key = f'k_{mode_idx}_LJ_{param_str}'
        
        # Read off the fields and abscissa
        field = fields[key]
        position = abscissa[key]
                    
        self.em_field_int.plot_fields(position, field,
                                      fname, cmap_str='jet',
                                      plot_scale='linear',
                                      plot_vector=plot_vector,
                                      transparent=True,
                                      plot_single_tet=plot_single_tet)
    
    def solve_AB_matrix_eqs(self):
        """
        Solves for the A, B matrices from the linear matrix equations
        """
        # Compute the field integrals
        ## Compute the E-fields
        self.em_field_int.filename_format = self.efield_format
        self.em_field_int.load_fields()
        print('Computing Akkp ...')
        Akkp = self.em_field_int.overlap_matrix_points(fname=None, cidxs=None,
                                      drop_points=1, return_matrix=True) 
        
        ## Compute the H-fields
        print('Computing Bkkp ...')
        self.em_field_int.filename_format = self.hfield_format
        self.em_field_int.load_fields()
        Bkkp = self.em_field_int.overlap_matrix_points(fname=None, cidxs=None,
                                      drop_points=1, return_matrix=True) 

        ## Compute the junction participations pL, pC
        print('Computing the junction participations ...')
        pL, pC = self.junc_pratios.compute_pL_pC(return_matrix = True)
        
        ## Repair the A, B matrices -- matrices of field overlaps, only
        print('Repairing Akkp, Bkkp ...')
        # Akkp, Bkkp = self.repair_ab_matrices(Akkp, Bkkp)
        
        # Iterate over inductances to compute the matrices at each node
        # for a given LJ, LJ0 pair
        LJs = self.LJs
        NLJ = LJs.size
        Nmodes = Akkp.shape[-1]
        AE = np.zeros_like(Akkp)
        AH = np.zeros_like(Bkkp)
        XE = np.zeros_like(Akkp)
        XH = np.zeros_like(Bkkp)
        II = np.eye(self.Nmodes)

        ## Solve the system of equations for AE, AH
        print('Solving the AE, AH systems of equations ...')
        for LJidx, LJ in enumerate(LJs):
            for LJpidx, LJp in enumerate(LJs):
                for k in range(self.Nmodes):
                    for kp in range(self.Nmodes):
                        if self.use_junction_participations:
                            # Use 1 / sqrt(pC_K pC_k'), 1 / sqrt(pL_k pLk')
                            XEfac = II[k, kp] - pC[LJpidx, k, kp]
                            XHfac = II[k, kp] - pL[LJpidx, k, kp]
                        else:
                            XEfac = 1
                            XHfac = 1
                        # Matrix inverses, should be relatively easy to invert
                        XE[LJpidx, LJpidx, k, kp] = XEfac * np.copy(Akkp[LJpidx, LJpidx, k, kp])
                        XH[LJpidx, LJpidx, k, kp] = XHfac * np.copy(Bkkp[LJpidx, LJpidx, k, kp])
                        
                # Compute the A, B coefficients by multiplying the matrix
                # of field overlaps by product of AE, AH with the inverses of XE, XH
                if self.use_junction_participations:
                    XEinv = np.linalg.inv(XE[LJpidx, LJpidx, :, :])
                    XHinv = np.linalg.inv(XE[LJpidx, LJpidx, :, :])
                else:
                    XEinv = II 
                    XHinv = II
                    
                AE[LJidx, LJpidx, :, :] = Akkp[LJidx, LJpidx, :, :] * XEinv
                AH[LJidx, LJpidx, :, :] = Bkkp[LJidx, LJpidx, :, :] * XHinv
        
        return AE, AH, XE, XH

class ParametricCouplings:
    """
    Compute the parametric coupling rates with the field integrals as inputs
    """
    def __init__(self, AE, AH, XE, XH, *args, **kwargs):
        """
        Start by loading the integrals into the class object, then running
        sums over the integrals to compute the coupling rates
        """
        # Standard class initialization as members from arguments and keyword
        # arguments, i.e. absorb all inputs into the class
        self.eprh = None
        self.Nmodes_in = None
        self.vstr = None
        self.vunits = None
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # Dictionary to convert between unit strings and scalings
        self.units = {'Hz' : 1., 'kHz' : 1e3, 'MHz' : 1e6, 'GHz' : 1e9}
        
        # Compute the number of modes from the dimensions of the inputs
        self.Nmodes = self.AE.shape[-1]
        self.NLJ = self.AE.shape[0]

        # Load the frequencies from HFSS
        self.load_wk_from_hfss(M=self.Nmodes_in)
    
    def load_wk_from_hfss(self, M : int = None):
        """
        Load resonane frequencies from HFSS EPR object, eprh
        """
        if self.eprh is not None:
            eprh = self.eprh
        else:
            raise KeyError('eprh is not set.')

        # Change the number of modes from the outside
        if M is not None:
            Nmodes = M
        else:
            Nmodes = self.Nmodes_in

        # Read off the inductances and x-offsets from the variations
        variations = eprh.solutions.list_variations()
        vstr = self.vstr      # 'cavity_offset'
        vunits = self.vunits  # 'mm'
        Ljstrs = []; Lj_vals = []

        #xoffstrs = []; xoff_vals = []
        for v in variations:
            Ljstrs.append(regex.findall(f'{vstr}=\'(.*?{vunits})', v)[0].replace('.', 'p'))
            Lj_vals.append(float(regex.findall(f'{vstr}=\'(.*?){vunits}', v)[0]))

        # Sort and reduce
        _, Lj_sort_idxs = np.unique(Lj_vals, return_index=True)
        Lj_vals = np.asarray(Lj_vals)[Lj_sort_idxs]
        Ljstrs = np.asarray(Ljstrs)[Lj_sort_idxs]
        vars_sorted = np.asarray(eprh.variations)
        vars_sorted = vars_sorted[Lj_sort_idxs]
        
        # for idx, v in enumerate(eprh.variations):
        wk = np.zeros([Nmodes, Nmodes, len(vars_sorted)])
        print(f'wk.shape: {wk.shape}')
        print(f'Ljstr.shape: {Ljstrs.shape}')
        print(f'Lj_sort_idxs.shape: {Lj_sort_idxs.shape}')
    
        for idx, v in enumerate(vars_sorted):
            # Set the variation
            eprh.set_variation(v)
            lv = eprh._get_lv(v)
            ww, qq = eprh.solutions.eigenmodes(lv=lv)
            wk[:, :, idx] = np.diag(ww[0:Nmodes])
        
        # Store the modes as internal class objects
        self.wk0 = np.copy(wk)
        
        return wk
        
    def get_w_gsms_k(self, k : int, units : str = 'MHz'):
        """
        Compute the resonance frequencies and single mode squeezing coupling rate
        in units of `units`
        """
        # Determine the units to return the couplings
        scale = self.units[units]
        
        # Get the XE, XH matrix elements
        Ek = self.XE
        Hk = self.XH
        
        # Get the AE, AH matrices
        AE, AH = self.AE, self.AH
        
        # Compute the sum naively
        Nmodes = self.Nmodes
        NLJ = self.NLJ
        AEsum = np.zeros([NLJ, NLJ])
        AHsum = np.zeros([NLJ, NLJ])
        for LJidx in range(NLJ):
            for LJpidx in range(NLJ):
                esum = 0.
                hsum = 0.
                for kpp in range(Nmodes):
                    AEprod = AE[LJidx, LJpidx, k, kpp]**2 * Ek[LJpidx, LJpidx, kpp, kpp]
                    AHprod = AH[LJidx, LJpidx, k, kpp]**2 * Hk[LJpidx, LJpidx, kpp, kpp]
                    # AEsum[LJidx, LJpidx] += AEprod * Ek[LJpidx, LJpidx, kpp, kpp]
                    # AHsum[LJidx, LJpidx] += AHprod * Hk[LJpidx, LJpidx, kpp, kpp]
                    esum += AEprod
                    hsum += AHprod
                AEsum[LJidx, LJpidx] = esum # AEprod # * Ek[LJidx, LJidx, kpp, kpp]
                AHsum[LJidx, LJpidx] = hsum # AHprod # * Hk[LJidx, LJidx, kpp, kpp]
                
        # Compute the OmegaE, OmegaH terms
        # OmegaEk = 0.5 * (sc.epsilon_0 / sc.h) * AEsum
        # OmegaHk = 0.5 * (sc.mu_0 / sc.h) * AHsum
        wk0 = self.wk0[k, k, :]
        OmegaEk = 0.25 * wk0 * AEsum
        OmegaHk = 0.25 * wk0 * AHsum
        
        # Compute the mode frequencies
        wk = 0.5 * (OmegaHk + OmegaEk)
        gsmsk = OmegaHk - OmegaEk
        
        return np.diag(wk) / scale, gsmsk / scale
    
    def get_gbs_gtms_kkp(self, k : int, kp : int, units : str = 'MHz'):
        """
        Compute the two-mode squeezing and beam splitter coupling rates
        """
        # Determine the units to return the couplings
        scale = self.units[units]
        
        # Get the XE, XH matrix elements
        Ek = self.XE
        Hk = self.XH
        
        # Get the AE, AH matrices
        AE, AH = self.AE, self.AH
        
        # Compute the sum naively
        Nmodes = self.Nmodes
        NLJ = self.NLJ
        AEsum = np.zeros([NLJ, NLJ])
        AHsum = np.zeros([NLJ, NLJ])
        for LJidx in range(NLJ):
            for LJpidx in range(NLJ):
                esum = 0.
                hsum = 0.
                for kpp in range(Nmodes):
                    AEprod = AE[LJidx, LJpidx, k, kpp] * AE[LJidx, LJpidx, kp, kpp] * Ek[LJpidx, LJpidx, kpp, kpp]
                    AHprod = AH[LJidx, LJpidx, k, kpp] * AH[LJidx, LJpidx, kp, kpp] * Hk[LJpidx, LJpidx, kpp, kpp]
                    # AEsum[LJidx, LJpidx] += AEprod * Ek[LJpidx, LJpidx, kpp, kpp]
                    # AHsum[LJidx, LJpidx] += AHprod * Hk[LJpidx, LJpidx, kpp, kpp]
                    esum += AEprod
                    hsum += AHprod
                AEsum[LJidx, LJpidx] = esum # * Ek[LJidx, LJidx, kpp, kpp]
                AHsum[LJidx, LJpidx] = hsum # * Hk[LJidx, LJidx, kpp, kpp]
                    
        # Compute the OmegaE, OmegaH terms
        # OmegaEk = 0.5 * (sc.epsilon_0 / sc.h) * AEsum
        # OmegaHk = 0.5 * (sc.mu_0 / sc.h) * AHsum
        wk0 = self.wk0[k, k, :]
        OmegaEk = 0.25 * wk0 * AEsum
        OmegaHk = 0.25 * wk0 * AHsum
        
        # Compute the mode frequencies
        gbskkp = OmegaHk + OmegaEk
        gtmskkp = OmegaHk - OmegaEk
        
        return gbskkp, gtmskkp