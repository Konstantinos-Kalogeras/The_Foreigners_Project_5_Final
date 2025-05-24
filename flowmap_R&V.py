import numpy as np 
import matplotlib.pyplot as plt 
import xarray as xr
import matplotlib.ticker as ticker 
import pickle
from py_wake import HorizontalGrid 
from py_wake.utils.plotting import setup_plot
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site._site import UniformWeibullSite, PowerShear
from py_wake.literature.gaussian_models import (Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, 
                                                Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018, Blondel_Cathelain_2020)
from py_wake.literature import Nygaard_2022
from py_wake.turbulence_models import CrespoHernandez 


with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Changfang_boundary.pkl", 'rb') as f:
    boundary1 = np.array(pickle.load(f))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Changfang_layout.pkl", 'rb') as f:
    xinit1,yinit1 = np.array(pickle.load(f))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Taipay_boundary.pkl", 'rb') as g:
    boundary2 = np.array(pickle.load(g))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Taipay_layout.pkl", 'rb') as g:
    xinit2,yinit2 = np.array(pickle.load(g))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Zhongneng_boundary.pkl", 'rb') as g:
    boundary3 = np.array(pickle.load(g))

with open(r"C:/Users/trinh/software_engin480/PyWake/The_Foreigners_Project_5_main/Zhongneng_layout.pkl", 'rb') as g:
    xinit3,yinit3 = np.array(pickle.load(g)) 


# All the farms use the same models (i.e., taipay, Zhongneng, Changfang)
class V174_9(GenericWindTurbine): 
    def __init__(self):
        GenericWindTurbine.__init__(self, name='V174-9.5', diameter=174, hub_height=100, 
                                    power_norm=9500, turbulence_intensity=0.07) #intensity varies from 6-8%

class Taiwan_straight(UniformWeibullSite): #Revwind site
    def __init__(self, ti= 0.07, shear=PowerShear(h_ref=100, alpha = 0.1)):
        f = [12.5995, 46.5254, 2.4446, 1.3694, 1.2476, 2.1822,  11.1784, 11.3762, 4.1769, 1.9177,  1.9943,  2.9878]
        a = [ 7.69, 15.46, 3.55, 1.89, 1.97, 2.89, 8.85, 9.05, 5.94, 3.47, 3.05, 3.52]
        k = [ 1.455, 2.525, 1.131, 0.670, 1.271, 1.150, 1.998, 2.521, 2.119, 1.732, 1.756, 1.510]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit2, yinit2]).T
        self.name = 'Revwind'
        

# Define sites and turbine's location 
site1  = Taiwan_straight()
x1, y1 = xinit1, yinit1 
x2, y2 = xinit2, yinit2
x3, y3 = xinit3, yinit3
turbine1 = V174_9()

wdir = 30  # Wind direction
wsp = 9.71   # Wind speed


# Combine turbine's position & turbines 
X_full = np.concatenate([x1,x2,x3])
Y_full = np.concatenate([y1,y2,y3])

# Define the grid bounds based on your turbine positions, adding some margin
x_min, x_max = np.min(X_full) - 500, np.max(X_full) + 500
y_min, y_max = np.min(Y_full) - 500, np.max(Y_full) + 500

# Define grid points with desired resolution (e.g. 100 m step)
x_grid = np.arange(x_min, x_max, 50)
y_grid = np.arange(y_min, y_max, 50)

# Create HorizontalGrid
custom_grid = HorizontalGrid(x=x_grid, y=y_grid)

# Different Gaussian models
wf_model1 = Bastankhah_PorteAgel_2014(site1, turbine1, k = 0.0324555)

wf_model2 = Zong_PorteAgel_2020(site1, turbine1)

wf_model3 = Blondel_Cathelain_2020(site1, turbine1, turbulenceModel=CrespoHernandez())

wf_model4 = Nygaard_2022(site1, turbine1)


# Different Gaussian models
wf_model1 = Bastankhah_PorteAgel_2014(site1, turbine1, k = 0.0324555)

wf_model2 = Zong_PorteAgel_2020(site1, turbine1)

wf_model3 = Blondel_Cathelain_2020(site1, turbine1, turbulenceModel=CrespoHernandez())

wf_model4 = Nygaard_2022(site1, turbine1)


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

for ax in axs.flat:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y/1e6:.2f}M'))

plt.tight_layout()
plt.show()

