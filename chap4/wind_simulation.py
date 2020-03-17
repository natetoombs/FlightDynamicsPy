"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import transfer_function
import numpy as np
import parameters.wind_parameters as windy

class wind_simulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[windy.Vw_ss_n],
                                      [windy.Vw_ss_e],
                                      [windy.Vw_ss_d]])
        Va = 25

        a1 = windy.sig_u * np.sqrt(2 * Va / (np.pi * windy.L_u))
        b1 = Va/windy.L_u

        a2 = windy.sig_v * np.sqrt(3 * Va / (np.pi * windy.L_v))
        a3 = Va/(np.sqrt(3)*windy.L_v)
        b2 = Va/windy.L_v

        a4 = windy.sig_w * np.sqrt(3 * Va / (np.pi * windy.L_w))
        a5 = Va/(np.sqrt(3)*windy.L_v)
        b3 = Va/windy.L_w

        # Dryden Gust Params:

        self.u_w = transfer_function(num=np.array([[a1]]),
                                     den=np.array([[1, b1]]),
                                     Ts=Ts)
        self.v_w = transfer_function(num=np.array([[a2, a3]]),
                                     den=np.array([[1, 2*b2, b2**2.0]]),
                                     Ts=Ts)
        self.w_w = transfer_function(num=np.array([[a4, a5]]),
                                     den=np.array([[1, 2*b3, b3**2.0]]),
                                     Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        gust = np.array([[0.], [0.], [0.]])
        return np.vstack((self._steady_state, gust))

