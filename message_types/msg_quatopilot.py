"""
msg_autopilot
    - messages type for input to the autopilot
    
part of mavsim_python
    - Beard & McLain, PUP, 2012
    - Last update:
        2/5/2019 - RWB
"""
import numpy as np

class msg_quatopilot:
    def __init__(self):
        self.p_ref = np.array([0.0, 0.0, 0.0])  # commanded position
        self.q_ref = np.array([0.0, 0.0, 0.0, 1.0])  # commanded course angle in rad
        self.u_ref = 0.0  # commanded speed
