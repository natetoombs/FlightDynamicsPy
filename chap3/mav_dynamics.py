"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/17/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

#import parameters
import parameters.aerosonde_parameters as MAV

from tools.tools import Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:

        self._state = np.array([MAV.pn0, MAV.pe0, MAV.pd0, MAV.u0, MAV.v0, MAV.w0, MAV.e0, MAV.e1, MAV.e2, MAV.e3, MAV.p0, MAV.q0, MAV.r0
                                ])
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, forces_moments):
        '''

            Integrate the differential equations defining dynamics. 
            Inputs are the forces and moments on the aircraft.
            Ts is the time step between function calls.
        '''
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state = self._state + time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6] = self._state.item(6)/normE
        self._state[7] = self._state.item(7)/normE
        self._state[8] = self._state.item(8)/normE
        self._state[9] = self._state.item(9)/normE

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        p_dot = np.array([
                         [e1**2+e0**2-e2**2-e3**2, 2*(e1*e2-e3*e0), 2*(e1*e3+e2*e0)],
                         [2*(e1*e2+e3*e0), e2**2+e0**2-e1**2-e3**2, 2*(e2*e3-e1*e0)],
                         [2*(e1*e3-e2*e0), 2*(e2*e3+e1*e0), e3**2+e0**2-e1**2-e2**2]]) @ np.array([u, v, w])

        # position dynamics
        uvw_dot = (np.array([r*v-q*w,
                            p*w-r*u,
                            q*u-p*v]) + 1/MAV.mass*np.array([fx, fy, fz]))

        # rotational kinematics
        quat_dot = 1/2*np.array([[0, -p, -q, -r],
                                 [p, 0, r, -q],
                                 [q, -r, 0, p],
                                 [r, q, -p, 0]]) @ np.array([e0, e1, e2, e3])

        # rotational dynamics
        pqr_dot = np.array([
                            MAV.gamma1*p*q-MAV.gamma2*q*r+MAV.gamma3*l+MAV.gamma4*n,
                            MAV.gamma5*p*r-MAV.gamma6*(p**2-r**2)+1/MAV.Jy*m,
                            MAV.gamma7*p*q-MAV.gamma1*q*r+MAV.gamma4*l+MAV.gamma8*n]).T

        # collect the derivative of the states
        x_dot = np.hstack([p_dot, uvw_dot, quat_dot, pqr_dot])

        return x_dot

    def _update_msg_true_state(self):
        # update the true state message:
        phi, theta, psi = Quaternion2Euler(self._state[6],
                                           self._state[7],
                                           self._state[8],
                                           self._state[9])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
