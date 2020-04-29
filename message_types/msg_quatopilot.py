"""
msg_quatopilot
    - messages type for input to the quatopilot
    
"""
import numpy as np

class msg_quatopilot:
    def __init__(self):
        self.p_ref = np.array([0.0, 0.0, 0.0])  # commanded position
        self.q_ref = np.array([0.0, 0.0, 0.0, 1.0])  # commanded course angle in rad
        self.u_ref = 0.0  # commanded speed
