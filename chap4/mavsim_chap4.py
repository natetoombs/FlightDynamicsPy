"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots


# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    #-------set control surfaces-------------
    # delta_e = -0.45
    # delta_t = 1
    # delta_a = 0.00
    # delta_r = 0.000

    if sim_time < 0.2 * SIM.end_time:
        delta_e = -.3
        delta_t = .667
        delta_a = 0
        delta_r = 0
    elif  sim_time < 0.4*SIM.end_time:
        delta_e = -0.35
        delta_t = 1.0
        delta_a = .005
        delta_r = 0.05
    elif  sim_time < 0.5*SIM.end_time:
        delta_e = -0.5
        delta_t = 1
        delta_a = 0
        delta_r = -0.0


    delta = np.array([[delta_a, delta_e, delta_t, delta_r]]).T  # transpose to make it a column vector

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation



