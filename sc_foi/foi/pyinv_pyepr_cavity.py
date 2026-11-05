#!/usr/bin/env python
# coding: utf-8
"""
Minimum requirements to run PyInventor, pyEPR
"""

# import slab 
# from slab import *
import PyInventor
from PyInventor import pyinvent as pyinv
import numpy as np
import math, glob, time, os
import pandas as pd
import pyEPR
from pyEPR import ansys as HFSS
import scipy
from scipy import constants as const
from pint import UnitRegistry
ureg = UnitRegistry()
Q = ureg.Quantity


# # Inventor
# In[3]:

'''
In this demo we show how to make lattice of cavities
'''

class coax_cavity(object):
    """
    Implementation of the commands needed to model a coaxial cavity
    Includes the following features:
    * Geometry setup with PyInventor
    * HFSS analysis and energy participation quantization analysis with pyEPR

    """
    def __init__(self, *args, **kwargs):

        # Internal class members
        self._stock_parameters = None

        # Start by closing all parts in Inventor after launch
        pyinv.com_obj().close_all_parts()
        
        # Set document units
        self.units = 'metric'
        in2mm = 25.4
        
        # Set it to overwrite file every time the part is instantiated 
        overwrite = True

        # Set filename
        fname = 'Multipin_coax_Demo.ipt'
        path = ''

        # Setup part -- makes a new Inventor part
        self.part = PyInventor.iPart(path=path, prefix=fname, units=self.units,
                                overwrite=overwrite)
        
        # Set view as shaded with edges
        self.part.set_visual_style(shaded=True, edges=True, hidden_edges=False)

        # Call the setters with default values for now
        ## TODO: modify to accept user supplied values
        setattr(self, 'stock_parameters', None)

        # Setup the workplanes and sketches
        self.setup_workplanes_sketches()

        # Start by drawing the stock, i.e. the unmodified block
        self.draw_stock()

        # Next, draw the cavity and posts
        self.draw_cavity_and_posts()


    """
    Getters for class properties
    """
    @property
    def stock_parameters(self):
        return self._stock_parameters

    """
    Deleters for class properties
    """
    @stock_parameters.deleter
    def stock_parameters(self):
        del self._stock_parameters

    """
    Setters for class properties
    """
    @stock_parameters.setter
    def stock_parameters(self, params_dict=None):
        # Conversion factor to handle metric / imperial interface
        if self.units == 'metric':
            convfac = self.in2mm
        elif self.units == 'imperial':
            convfac = 1.0
        else:
            raise ValueError(f'Unsupported units ({self.units})')

        # Default values
        cav_width = 0.6
        cav_corner_rad = 0.25
        pin_dia = 0.125
        pin_height = 0.35
        pin_spacing = 0.2
        cav_depth = 1.75
        cav_edge_len = cav_width - 2.0*cav_corner_rad
        
        # Square lattice dimension
        side_cav_num = 1
        
        # Stock clearances
        cav_spacing = 0.125
        edge_spacing = 0.25
        stock_z_offset = 0.25

        # Save the cavity parameters to a dictionary
        self.cavity_parameters = {'cav_spacing' : cav_spacing, 
                                  'edge_spacing' : edge_spacing,
                                  'cav_width' : cav_width,
                                  'cav_depth' : cav_depth,
                                  'cav_corner_rad' : cav_corner_rad,
                                  'cav_edge_len' : cav_edge_len,
                                  'pin_spacing' : pin_spacing,
                                  'pin_origin' : pin_origin,
                                  'pin_dia' : pin_dia,
                                  'pin_height' : pin_height}
        
        # Setup stock
        stock_width = cav_width*side_cav_num + cav_spacing * (side_cav_num - 1)
                        + 2*edge_spacing
        stock_height = cav_depth + stock_z_offset
        stock_origin = (0.0, 0.0)

        # ----- Dictionary Setup -----
        ## Collect all of the variables as vals with corresponding keys
        keys = ['stock_width', 'stock_height', 'stock_origin', 'stock_z_offset']
        vals = [stock_width, stock_height, stock_origin, stock_z_offset]

        ## Store the dictionary variables
        key_cnt = sum([1 if k in keys for k in params.keys()])
        if (params is not None) and (key_cnt == len(keys)):
            self._stock_parameters = params
        else:
            self._stock_parameters = {k : v*convfac for k,v in zip(keys, vals)}


    def setup_workplanes_sketches(self):
        """
        Sets up the workplanes and sketches needed for the cavity
        """
        # Import stock parameters
        stock_params = self.stock_parameters

        # Setup workplanes -- Add a workplane to the part in the xy plane
        #                    offset by stock_z_offset on the z-axis
        stock_wp = self.part.add_workplane(plane='xy')
        cav_bottom_wp = self.part.add_workplane(plane='xy',
                            offset=stock_params['stock_z_offset'])

        # Setup sketches -- Add a new sketch to the workplane
        stock_sketch = self.part.new_sketch(stock_wp)
        cavity_sketch = self.part.new_sketch(cav_bottom_wp)

        # Save sketches and workplanes to class dictionaries
        self.sketches = {'stock' : stock_sketch, 'cavity' : cavity_sketch}
        self.workplanes = {'stock' : stock_wp, 'cavity_bottom' : cav_bottom_wp}


    def draw_stock(self):
        """
        Draw the stock for the cavity to be machined out of
        """
        # Get the sketches and stock parameters
        params = self.stock_params

        # Draw stock
        stock_base = pyinv.structure(self.part, self.sketches['stock'],
                               start=params['stock_origin'])
        stock_base.add_line(params['stock_width'], 180)
        stock_base.add_line(params['stock_width'], 90)
        stock_base.add_line(params['stock_width'], 0)
        
        stock_path = stock_base.draw_path(close_path=True)

        # Extrude stock -- Extrude the sketch defining the stock
        self.part.extrude(self.sketches['stock'],
                          thickness=params['stock_height'],
                          obj_collection=stock_path,
                          direction='positive', operation='join')
        
        # Fits whole cavity in frame
        self.part.view.Fit()

        # Make the stock sketch visible for editing later
        self.sketches['stock'].sketch_obj.Visible=True


    def draw_cavity_and_posts(self):
        """
        Draws the cavity by acting on the existing stock geometry
        """
        # Create empty list that makes up all the 3D objects of the cavity for
        # later.
        cav_objs = []

        # Load the stock and cavity parameters
        stock_params   = self.stock_params
        cavity_params  = self.cavity_params
        stock_origin   = stock_params['stock_origin']
        edge_spacing   = cavity_params['edge_spacing']
        cav_corner_rad = cavity_params['cav_corner_rad']
        cav_edge_len   = cavity_params['cav_edge_len']
        cav_depth      = cavity_params['cav_depth']
        cav_spacing    = cavity_params['cav_spacing']
        
        # Make the stock sketch visable for editing
        stock_sketch.sketch_obj.Visible = True
        
        # Draw cavity shape
        cav_origin = (stock_origin[0] - edge_spacing,
                      stock_origin[1] + edge_spacing + cav_corner_rad)
        cavity_base = pyinv.structure(self.part, self.sketches['cavity'],
                                      start=cav_origin)
        cavity_base.add_line(cav_edge_len, 90)
        cavity_base.add_line_arc(start_angle=0, stop_angle=90,
                                 radius=cav_corner_rad, flip_dir=True,
                                 rotation=0)
        cavity_base.add_line(cav_edge_len, 180)
        cavity_base.add_line_arc(start_angle=90, stop_angle=180,
                                 radius=cav_corner_rad, flip_dir=True,
                                 rotation=0)
        cavity_base.add_line(cav_edge_len, 270)
        cavity_base.add_line_arc(start_angle=0, stop_angle=270,
                                 radius=cav_corner_rad, flip_dir=True,
                                 rotation=0)
        cavity_base.add_line(cav_edge_len, 0)
        cavity_base.add_line_arc(start_angle=90, stop_angle=0,
                                 radius=cav_corner_rad, flip_dir=True,
                                 rotation=0)
        
        cav_base = cavity_base.draw_path(close_path=True)
        
        # Extrude and add to cav_objs list for later patterning
        cav_objs.append(self.part.extrude(self.sketches['cavity'],
                        thickness=cav_depth,
                        obj_collection=cav_base, direction='positive',
                        operation='cut'))

        # Post origin (center of post pattern)
        post_origin = (stock_origin[0] - edge_spacing - cav_width / 2,
                       stock_origin[1] + edge_spacing + cav_width / 2)
        
        # Create a set of points to define the 3 post centers based on the
        # spacing and the offset. Equilateral triangle with sides=.25
        post_rot_offset = 90
        rot_rad         = post_spacing/np.sqrt(3)

        # Circurlar pattern
        post_pts = pyinv.circle_pattern(radius=rot_rad, center_pt=post_origin,
                                  segments=2, offset=post_rot_offset)
        
        # Create sketches circles for posts and add to object
        post_obj=[]
        for pts in post_pts:
            post_obj.append(self.part.sketch_circle(self.sketches['cavity'],
                            center=pts, radius=post_dia/2))
        
        # Because we did one extrude operation we need to make sure the sketch
        # is visible before we try to extrude the posts
        cavity_sketch.sketch_obj.Visible = True
        
        #extrude posts
        for posts in post_obj:
            cav_objs.append(self.part.extrude(self.sketches['cavity'],
                            thickness=post_height, obj_collection=posts,
                            direction='positive', operation='join'))
        
        # Create an object collection from the above list of extruded features
        # that make up a single cell
        cav_obj_coll = self.part.create_obj_collection(cav_objs)
        
        # pattern feature from the above cavity object collection
        x_cav_feat = self.part.rectangular_feature_pattern(
                obj_collection=cav_obj_coll,
                count=side_cav_num, spacing=cav_spacing+cav_width, axis='x',
                direction='negative', fit_within_len=False)
        
        # Add the x new patterned features to the object collection and then
        # repeat the above, now with the updated collection and in the +y
        # direction
        cav_obj_coll.Add(x_cav_feat)
        y_cav_feat = self.part.rectangular_feature_pattern(
                     obj_collection=cav_obj_coll, count=side_cav_num,
                     spacing=cav_spacing+cav_width, axis='y',
                     direction='positive', fit_within_len=False)
        
        # Turn off the cavity_sketch visability
        self.sketches['cavity'].sketch_obj.Visible=False
        self.sketches['stock'].sketch_obj.Visible=False


    def save_part(self, copy_name=None):
        """
        Saves part to file
        """
        # Set final view to GoHome for stylistic effect
        time.sleep(2)
        self.part.view.GoHome()
        
        # Save document 
        self.part.save()

        # Save copy as stp file for export to HFSS
        self.path = self.part.save_copy_as(copy_name='coax_test.stp' )


# # Eigenmode simulation 

# In[3]:


'''
HFSS eigenmode simulation Creation:

This sets up a standard eigenmode simulation without the qubit, just the bare cavity created above. It calculates
the mode frequencies, loss (with boundary impedances set) and the electric and magnetic surface participation 
ratios (S_e, S_m)

'''

"""
project_name='Multipost_Coax_3'
design_name='Multipost_Coax_2'
overwrite=True

#use file location path:
HFSS_path=os.getcwd()

full_path=HFSS_path+'\\'+project_name+'.aedt'
HFSS_app=HFSS.HfssApp()
HFSS_desktop=HFSS_app.get_app_desktop()
project=HFSS_desktop.open_project(full_path)

if project==None:
    project=HFSS_desktop.new_project()
    project.save(full_path)
    
project.make_active()
    
if design_name in project.get_design_names():
    if overwrite==True:
        project.delete_design(design_name)
        project.save()
        EM_design=project.new_em_design(design_name)
    else:
        EM_design=project.get_design(design_name)
        
else:
    EM_design=project.new_em_design(design_name)
    
    
EM_design.make_active()
model=HFSS.HfssModeler(EM_design)

# import inventor design
model.import_3D_obj(path)

#create variables for
Stock_L=EM_design.create_variable('Stock_L', '%.3fin'%(-stock_width))
Stock_W=EM_design.create_variable('Stock_W', '%.3fin'%(stock_width))
Stock_H=EM_design.create_variable('Stock_H', '%.3fin'%(stock_height))
Post_H=EM_design.create_variable('Post_H', '%.3fin'%(post_height))
Post_Sp=EM_design.create_variable('Post_Sp', '%.3fin'%(post_spacing))

cav_dims=[Stock_L, Stock_W, Stock_H]

# Makes box around part
box=model.draw_box_corner([0,0,0], cav_dims)
objs=model.get_object_names()
obj_name=model.subtract(blank_name=objs[1], tool_names=[objs[0]])

# Not sure why this does not work.
# top_id=model.get_face_id_by_pos(obj=obj_name, pos=[0,0,Stock_H])

top_id=int(model.get_face_ids(obj = obj_name)[-1])
model.assign_impedance(377, 0, obj_name, top_id)


EM_setup=EM_design.create_em_setup(name="Test_EM", min_freq_ghz=2, n_modes=3, max_delta_f=0.1)
EM_setup.analyze()
solns=EM_setup.get_solutions()

eigen_real=solns.eigenmodes()[0]
eigen_imag=solns.eigenmodes()[1]


calc=HFSS.HfssFieldsCalc(EM_setup)

calc.clear_named_expressions()

#be sure to change path to wher ethe S_v and S_m calcs are
Se_calc_path='S:\\Andrew Oriani\\CQED 3D Resonators\\HFSS Calculators\\E_energy_S_to_V.clc'
Sm_calc_path='S:\\Andrew Oriani\\CQED 3D Resonators\\HFSS Calculators\\H_energy_S_to_V.clc'

Se_name=calc.load_named_expression(Se_calc_path)
Sm_name=calc.load_named_expression(Sm_calc_path)

print('Eigenmodes:')
print('_____________________________')
for n, (em_real, em_imag) in enumerate(zip(eigen_real, eigen_imag)):
    print('Mode #%i: %.3f+i%.3e GHz'%(n+1, em_real, em_imag))

print('_____________________________')

n_modes=2
Se=[]
Sm=[]
for n in range(1,n_modes+1):
    solns.set_mode(n)
    Se.append(calc.use_named_expression(Se_name).evaluate())
    Sm.append(calc.use_named_expression(Sm_name).evaluate())
    print('Se=%.3f, Sm=%.3f for mode number %i'%(Se[-1], Sm[-1], n))

n_modes=3
Se=[]
Sm=[]
for n in range(1,n_modes+1):
    solns.set_mode(n)
    Se.append(calc.use_named_expression(Se_name).evaluate())
    Sm.append(calc.use_named_expression(Sm_name).evaluate())
    print('Se=%.3f, Sm=%.3f for mode number %i'%(Se[-1], Sm[-1], n))
"""
