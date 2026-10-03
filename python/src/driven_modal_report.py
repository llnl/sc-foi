# ----------------------------------------------
# Script Recorded by ANSYS Electronics Desktop Version 2020.2.0
# 8:34:58  Jul 07, 2022
# ----------------------------------------------
import ScriptEnv

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject("planar_pads_l_coupled_three_geom2_cavities_bbq_y11_220707")
oDesign = oProject.SetActiveDesign("planar_pads_l_coupled_three_geom2_cavities_bbq_y11_hfss_design_220707")
oModule = oDesign.GetModule("ReportSetup")
oModule.CreateReport("Y Parameter Plot 1", "Modal Solution Data", "Rectangular Plot", "Setup : Sweep", 
	[
		"Domain:="		, "Sweep"
	], 
	[
		"Freq:="		, ["All"],
		"z_cav_1:="		, ["Nominal"],
		"y_cav_1:="		, ["Nominal"],
		"x_cav_1:="		, ["Nominal"],
		"z_cav_2:="		, ["Nominal"],
		"y_cav_2:="		, ["Nominal"],
		"x_cav_2:="		, ["Nominal"],
		"z_coupler:="		, ["Nominal"],
		"y_coupler:="		, ["Nominal"],
		"x_coupler:="		, ["Nominal"],
		"ycoupler_offset:="	, ["Nominal"],
		"zcoupler_offset:="	, ["Nominal"],
		"z_offset:="		, ["Nominal"],
		"x_cpl_offset:="	, ["Nominal"],
		"cav_thickness:="	, ["Nominal"],
		"coupler_pads:="	, ["Nominal"],
		"y_pads:="		, ["Nominal"],
		"l_junc:="		, ["Nominal"],
		"x_sub:="		, ["Nominal"],
		"y_sub:="		, ["Nominal"],
		"z_sub:="		, ["Nominal"],
		"Lj_1:="		, ["Nominal"]
	], 
	[
		"X Component:="		, "Freq",
		"Y Component:="		, ["re(Y(LumpPort,LumpPort))","im(Y(LumpPort,LumpPort))"]
	])
oModule.ExportToFile("Y Parameter Plot 1", "Z:/tcav/data/y11_sapphire_three_cavities_planar_plates_long_coupler_5_20GHz_220707.csv", False)
