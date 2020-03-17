"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap3.mav_dynamics import mav_dynamics


# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if VIDEO == True:
    from chap2.video_writer import videoWriter
    video = videoWriter(video_name="chap3_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    fx = 1.5
    fy = 0
    fz = 0
    Mx = 0
    My = 0
    Mz = 0.001
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # #-------vary forces and moments to check dynamics-------------
    # if sim_time < 0.1*SIM.end_time:
    #     fx = 10
    #     fy = 0
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.2*SIM.end_time:
    #     fx = -10
    #     fy = 0
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.3*SIM.end_time:
    #     fx = 0
    #     fy = 10
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.4*SIM.end_time:
    #     fx = 0
    #     fy = -10
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.5*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = -10
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.6*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 10
    #     Mx = 0.0
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.65*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.3
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.75*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = 0.0
    #     My = -0.3
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.8*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = 0.0
    #     My = 0.3
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < 0.85*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = 0.1
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < .95*SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = -0.1
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # elif sim_time < SIM.end_time:
    #     fx = 0
    #     fy = 0
    #     fz = 0
    #     Mx = 0.1
    #     My = 0.0
    #     Mz = 0.0
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    #-------physical system-------------
    mav.update_state(forces_moments)  # propagate the MAV dynamics
    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAVs
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




