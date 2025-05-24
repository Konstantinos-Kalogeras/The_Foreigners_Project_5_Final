import numpy as np 
import matplotlib.pyplot as plt 
import xarray as xr
from py_wake import HorizontalGrid 
import pickle
from py_wake.utils.plotting import setup_plot
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site._site import UniformWeibullSite, PowerShear
from py_wake.literature.gaussian_models import (Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, 
                                                Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018, Blondel_Cathelain_2020)
                                                
from py_wake.literature import Nygaard_2022
from py_wake.rotor_avg_models import RotorCenter
from py_wake.turbulence_models import CrespoHernandez 


with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5/Vinyard_utm_boundary.pkl", 'rb') as f:
    boundary1 = np.array(pickle.load(f))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Vinyard_utm_layout.pkl", 'rb') as f:
    xinit1,yinit1 = np.array(pickle.load(f))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5/Revolution_utm_boundary.pkl", 'rb') as g:
    boundary2 = np.array(pickle.load(g))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5/Revolution_utm_layout.pkl", 'rb') as g:
    xinit2,yinit2 = np.array(pickle.load(g))


class SG_11200(GenericWindTurbine): #Revwind Turbine
    def __init__(self):
        GenericWindTurbine.__init__(self, name='SG 11-200', diameter=200, hub_height=100, 
                                    power_norm=11000, turbulence_intensity=0.07) #intensity varies from 6-8%


class RevolutionWindData(UniformWeibullSite):
    def __init__(self, ti= 0.07, shear=PowerShear(h_ref=100, alpha = 0.1)):
        f = [6.5294, 7.4553, 6.2232, 5.8886, 4.7439, 4.5632, 7.1771, 12.253, 13.8541, 10.3711, 11.5819, 9.3593]
        a = [9.93, 10.64, 9.87, 8.85, 8.46, 8.26, 10.45, 11.75, 11.40, 10.82, 11.95, 10.08]
        k = [2.385, 1.822, 1.979, 1.842, 1.607, 1.486, 1.865, 2.256, 2.678, 2.170, 2.455, 2.506]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit2, yinit2]).T
        self.name = "Revolution South Fork Wind"


class Haliade_X(GenericWindTurbine): #Vineyard Turbine
    def __init__(self):
        GenericWindTurbine.__init__(self, name='Haliade-X', diameter=220, hub_height=150, 
                                    power_norm=13000, turbulence_intensity=0.07) #intensity varies from 6-8%

class VinyardWind2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=PowerShear(h_ref=150, alpha=0.1), wd = 270):
        f = [6.4452, 7.6731, 6.4753, 6.0399, 4.8786, 4.5063, 7.318, 11.7828, 13.0872, 11.1976, 11.1351, 9.461]
        a = [10.26, 10.44, 9.52, 8.96, 9.58, 9.72, 11.48, 13.25, 12.46, 11.40, 12.35, 10.48]
        k = [2.225, 1.697, 1.721, 1.689, 1.525, 1.498, 1.686, 2.143, 2.369, 2.186, 2.385, 2.404]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit1, yinit1]).T
        self.name = "Vineyard Wind Farm"


# Define sites and turbine's location 
site1, site2  = VinyardWind2(), RevolutionWindData()
x1, y1 = site1.initial_position.T
x2, y2 = site2.initial_position.T
turbine1, turbine2 = Haliade_X(), SG_11200()

wdir = 270  # Wind direction
wsp = 9.76   # Wind speed


# Combine turbine's position & turbines 
X_full = np.concatenate([x1, x2])
Y_full = np.concatenate([y1, y2])
combined_turbines = WindTurbines.from_WindTurbine_lst([turbine1, turbine2])

# Define the grid bounds based on your turbine positions, adding some margin
x_min, x_max = np.min(X_full) - 500, np.max(X_full) + 500
y_min, y_max = np.min(Y_full) - 500, np.max(Y_full) + 500

# Define grid points with desired resolution (e.g. 100 m step)
x_grid = np.arange(x_min, x_max, 100)
y_grid = np.arange(y_min, y_max, 100)

# Create HorizontalGrid
custom_grid = HorizontalGrid(x=x_grid, y=y_grid)

# Different Gaussian models
wf_model1 = Bastankhah_PorteAgel_2014(site1, combined_turbines, k = 0.0324555)

wf_model2 = Zong_PorteAgel_2020(site1, combined_turbines)

wf_model3 = Blondel_Cathelain_2020(site1, combined_turbines, turbulenceModel=CrespoHernandez())

wf_model4 = Nygaard_2022(site1, combined_turbines)


# Different Simulation 
sim_res1 = wf_model1(X_full, Y_full, wd = [wdir], ws = [wsp])

sim_res2 = wf_model2(X_full, Y_full, wd = [wdir], ws = [wsp])

sim_res3 = wf_model3(X_full, Y_full, wd = [wdir], ws = [wsp])

sim_res4 = wf_model4(X_full, Y_full, wd = [wdir], ws = [wsp])


# Differnet Flow maps
flow_map1 = sim_res1.flow_map(grid=None)

flow_map2 = sim_res2.flow_map(grid=None)

flow_map3 = sim_res3.flow_map(grid=None)

flow_map4 = sim_res4.flow_map(grid=custom_grid) 


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a grid of Axes

# Plot 1
flow_map1.plot_wake_map(levels=10, cmap='jet', plot_colorbar=True,
                        plot_windturbines=False, ax=axs[0, 0])
axs[0, 0].set_title("Bastankhah")
axs[0, 0].set_xlabel("x [m]")
axs[0, 0].set_ylabel("y [m]")
axs[0, 0].axis("auto")

# Plot 2
flow_map2.plot_wake_map(levels=10, cmap='jet', plot_colorbar=True,
                         plot_windturbines=False, ax=axs[0, 1])
axs[0, 1].set_title("Zong")
axs[0, 1].set_xlabel("x [m]")
axs[0, 1].set_ylabel("y [m]")
axs[0, 1].axis("auto")

# Plot 3
flow_map3.plot_wake_map(levels=10, cmap='jet', plot_colorbar=True,
                        plot_windturbines=False, ax=axs[1, 0])
axs[1, 0].set_title("SuperGaussian (Blondel_Cathelain2020)")
axs[1, 0].set_xlabel("x [m]")
axs[1, 0].set_ylabel("y [m]")
axs[1, 0].axis("auto")

# Plot 4
flow_map4.plot_wake_map(levels=10, cmap='jet', plot_colorbar=True,
                         plot_windturbines=False, ax=axs[1, 1])
axs[1, 1].set_title("Nygaard")
axs[1, 1].set_xlabel("x [m]")
axs[1, 1].set_ylabel("y [m]")
axs[1, 1].axis("auto")

plt.tight_layout()
plt.show()

