import numpy as np
import topfarm
import matplotlib.pyplot as plt
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import (
    MultiWFBoundaryConstraint,
    BoundaryType,
)
from topfarm.constraint_components.constraint_aggregation import (
    DistanceConstraintAggregation,
)
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import (
    PyWakeAEPCostModelComponent,
)
from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySGDDriver
from topfarm.plotting import XYPlotComp
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.utils.gradients import autograd
from py_wake.validation.lillgrund import LillgrundSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from py_wake.site._site import UniformWeibullSite, PowerShear
import pickle
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint

# Extracting coordinatates of wind turbines and boundarys
#RESPECT PATH DIRECTORIES
#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Vinyard_utm_boundary.pkl  'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/boundaries/Changfang_boundary.pkl', 'rb') as f:
    boundary1 = np.array(pickle.load(f))

#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Vinyard_utm_layout.pkl  'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/tbl_layouts/Changfang_layout.pkl', 'rb') as f:
    xinit1,yinit1 = np.array(pickle.load(f))

#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Revolution_utm_boundary.pkl 'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/boundaries/Taipay_boundary.pkl', 'rb') as g:
    boundary2 = np.array(pickle.load(g))

#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Revolution_utm_layout.pkl 'Kons'
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/tbl_layouts/Taipay_layout.pkl', 'rb') as g:
    xinit2,yinit2 = np.array(pickle.load(g))

#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Taiwan_farms\Zhongneng_boundary.pkl
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/boundaries/Zhongneng_boundary.pkl', 'rb') as g:
    boundary3 = np.array(pickle.load(g))

#E:\Spring 2025\ENGIN 480\Project_5\The_Foreigners_Project_5\Taiwan_farms\Zhongneng_layout.pkl
with open(r'/Users/ompatel/Documents/Proj5_Engin480/The_Foreigners_Project_5-1/tbl_layouts/Zhongneng_layout.pkl', 'rb') as g:
    xinit3,yinit3 = np.array(pickle.load(g))    

#number of iterations for opt
maxiter = 1000
tol = 1e-6

##########################################################################################################################
class V174_9(GenericWindTurbine): # taipay, Zhongneng, Changfang Turbine's
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
        self.name = 'Revolution South Fork Wind'
##########################################################################################################################        
wt_tiwan = V174_9()
site_tiwan = Taiwan_straight()

# windfarm model for all sites
wf_model = Bastankhah_PorteAgel_2014( #used to be wf1_model
    site_tiwan,
    wt_tiwan,
    k=0.0324555,  # default value from BastankhahGaussianDeficit
)

wt_x1, wt_y1 = xinit1, yinit1

wt_x2, wt_y2 = xinit2, yinit2

wt_x3, wt_y3 = xinit3, yinit3

# Collecting the wind terbine locations into one list using numpy
X_full = np.concatenate([wt_x1, wt_x2, wt_x3])
Y_full = np.concatenate([wt_y1, wt_y2, wt_y3])

# Amount of wind turbines and print the number 
n_wt = len(X_full)
print(f"Initial layout has {n_wt} wind turbines")

# plot initial layout - shows each turbine numbered and tracked 
plt.figure()
plt.plot(X_full, Y_full, "x", c="magenta")
# put indeces on the wind turbines
for i in range(n_wt):
    plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
plt.axis("equal")
plt.show()

# print('done')
 
# masking Vinyard wind
n_wt_sf = n_wt - 61                 
# n_wt is the amount of turbines included in this figure there are 76 total turbines 
# 0-63 are vinyards we need to isolate them to mask them so 67-12 = 63

# Masking vinyard and costal
wf1_mask = np.zeros(n_wt, dtype=bool)
wf1_mask[:n_wt_sf] = True
wf2_mask = np.zeros(n_wt, dtype=bool)
wf2_mask[57:88] =  True

wf3_mask = ~(wf1_mask|wf2_mask)  # the rest of turbines

# printing which mask belongs to which wind farm 
print(f"Turbines belonging to wind farm 1: {np.where(wf1_mask)[0]}") # verifiing that our calulations were correct 
print(f"Turbines belonging to wind farm 2: {np.where(wf2_mask)[0]}")
print(f"Turbines belonging to wind farm 2: {np.where(wf3_mask)[0]}")

# putting the different wind farms in to two disktict list's 
wt_groups = [np.arange(0, n_wt-61), np.arange(n_wt-61, n_wt-30),np.arange(n_wt-30, n_wt)]


# defining the constraints 
constr_type = BoundaryType.POLYGON
constraint_comp = MultiWFBoundaryConstraint(geometry = [boundary1, boundary2, boundary3], wt_groups=wt_groups,
        boundtype=constr_type)

# let's see how the boundaries look like
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(X_full, Y_full, "x", c="magenta")
for i in range(n_wt):
    plt.text(X_full[i] + 10, Y_full[i], str(i + 1), fontsize=12)
plt.axis("equal")
constraint_comp.get_comp(n_wt).plot(ax1)
plt.show()


# AEP Cost Model Component - SLSQP
slsqp_cost_comp = PyWakeAEPCostModelComponent(
    windFarmModel=wf_model, n_wt=n_wt, grad_method=autograd
)


# defining the driver
# driver_type = "SGD"  
driver_type = "SLSQP"
min_spacing = wt_tiwan.diameter() * 2

# If SLSQP define use these constraints and spacing  
if driver_type == "SLSQP":
    constraints = [
        constraint_comp,
        SpacingConstraint(min_spacing=min_spacing),
    ]
    driver = EasyScipyOptimizeDriver(
        optimizer="SLSQP",
        maxiter=maxiter,
    )
    cost_comp = slsqp_cost_comp
                                            
# defining problem for optimizer calling both farms to be optimized at the same time
problem = TopFarmProblem(
    design_vars={"x": X_full, "y": Y_full},
    n_wt=n_wt,
    constraints=constraints,
    cost_comp=cost_comp,
    driver=driver,
    plot_comp=XYPlotComp(),
)

# Initialting recorder, AEP cost, and optimization problem
cost, state, recorder = problem.optimize(disp = True)


# Where to save the recording what the file should be names as 
# recorder.save('optimization_Vinyard_respecting_Revolution_2')
recorder.save('Optimization_tiwan_farms')

print('done')