"""
final_sim
    - Implementation of Quaternion-based Autopilot
    - Uncomment the desired control here, in
    - quatopilot.py, and in quat_parameters.py
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from finalProject.mav_dynamics import mav_dynamics
from chap4.wind_simulation import wind_simulation
from finalProject.quatopilot import quatopilot
from message_types.msg_quatopilot import msg_quatopilot
from tools.tools import Euler2Quaternion

# initialize the visualization
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots

# initialize elements of the architecture
wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)
ctrl = quatopilot(SIM.ts_simulation)

# # autopilot commands -- Hover (In development)
# commands = msg_quatopilot()
# commands.p_ref = np.array([0, 0, -100])
# rpy = np.array([0, 90, 0])
# commands.q_ref = Euler2Quaternion(np.radians(rpy[0]), np.radians(rpy[1]), np.radians(rpy[2]))
# commands.u_ref = 0

# autopilot commands -- Off Axis
commands = msg_quatopilot()
commands.p_ref = np.array([10, 0, -100])
rpy = np.array([15, 10, -10])
commands.q_ref = Euler2Quaternion(np.radians(rpy[0]), np.radians(rpy[1]), np.radians(rpy[2]))
commands.u_ref = 20

# # autopilot commands -- Straight and Level
# commands = msg_quatopilot()
# commands.p_ref = np.array([10, 0, -100])
# rpy = np.array([0, 30, 0])
# commands.q_ref = Euler2Quaternion(np.radians(rpy[0]), np.radians(rpy[1]), np.radians(rpy[2]))
# commands.u_ref = 20

# # autopilot commands -- Straight-Up
# commands = msg_quatopilot()
# commands.p_ref = np.array([40, 0, -110])
# rpy = np.array([0, 85, 0])
# commands.q_ref = Euler2Quaternion(np.radians(rpy[0]), np.radians(rpy[1]), np.radians(rpy[2]))
# commands.u_ref = 15

# initialize the simulation time
sim_time = SIM.start_time

# i = 0
# while i<50:
#     estimated_state = mav.msg_true_state
#     delta = np.array([0, 0, 0, 0])
#     current_wind = wind.update()  # get the new wind vector
#     mav.update_state(delta, current_wind)  # propagate the MAV dynamics
#     i += 1

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------controller-------------
    estimated_state = mav.msg_true_state  # uses true states in the control
    delta = ctrl.update(commands, estimated_state)

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, #commanded_state,    # commanded states
                     SIM.ts_simulation)

    #-------increment time-------------
    sim_time += SIM.ts_simulation
