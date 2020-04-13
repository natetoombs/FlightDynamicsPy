"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

"""
import sys

sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as mav
from tools.tools import Quaternion2Rotation, Quaternion2Euler


class mav_dynamics:
    def __init__(self, Ts):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        # self._state = np.array([[MAV.pn0],  # (0)
        #                        [MAV.pe0],   # (1)
        #                        [MAV.pd0],   # (2)
        #                        [MAV.u0],    # (3)
        #                        [MAV.v0],    # (4)
        #                        [MAV.w0],    # (5)
        #                        [MAV.e0],    # (6)
        #                        [MAV.e1],    # (7)
        #                        [MAV.e2],    # (8)
        #                        [MAV.e3],    # (9)
        #                        [MAV.p0],    # (10)
        #                        [MAV.q0],    # (11)
        #                        [MAV.r0]]).T   # (12)
        self._state = np.array(
            [mav.pn0, mav.pe0, mav.pd0, mav.u0, mav.v0, mav.w0, mav.e0, mav.e1, mav.e2, mav.e3, mav.p0, mav.q0, mav.r0
             ])
        # self.true_state = np.array([[MAV.pn0],  # (0)
        #                             [MAV.pe0],  # (1)
        #                             [MAV.pd0],  # (2)
        #                             [MAV.Va0],  # (3)
        #                             [MAV.alpha0],  # (4)
        #                             [0],  # (5) beta0?
        #                             [MAV.phi0],  # (6
        #                             [MAV.theta0],  # (7)
        #                             [0],  # (8) chi0
        #                             [MAV.p0],  # (9)
        #                             [MAV.q0],  # (10)
        #                             [MAV.r0],  # (11)
        #                             [MAV.Va0],  # (12)
        #                             [MAV.u0],  # (13)
        #                             [MAV.v0],  # (14)
        #                             [MAV.psi0],  # (15)
        #                             [0],  # (16)
        #                             [0],  # (17)
        #                             [0]])  # (18)

        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = mav.Va0
        self._Vg = self._Va
        self._alpha = 0
        self._beta = 0
        self.thrust = 0
        self._chi = 0
        self._Vs = self._Va
        # initialize true_state message
        self.msg_true_state = msg_state()
        self.msg_true_state.Vs = self._Vs

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, delta)
        k2 = self._derivatives(self._state + time_step / 2. * k1, delta)
        k3 = self._derivatives(self._state + time_step / 2. * k2, delta)
        k4 = self._derivatives(self._state + time_step * k3, delta)
        self._state = self._state + time_step / 6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0 ** 2 + e1 ** 2 + e2 ** 2 + e3 ** 2)
        self._state[6] = self._state.item(6) / normE
        self._state[7] = self._state.item(7) / normE
        self._state[8] = self._state.item(8) / normE
        self._state[9] = self._state.item(9) / normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()


    ###################################
    # private functions
    def _derivatives(self, state, delta):
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

        forces_moments = self._forces_moments(delta)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        p_dot = np.array([
            [e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
            [2 * (e1 * e2 + e3 * e0), e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2, 2 * (e2 * e3 - e1 * e0)],
            [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2]]) @ np.array(
            [u, v, w])

        # position dynamics
        uvw_dot = (np.array([r * v - q * w,
                             p * w - r * u,
                             q * u - p * v]) + 1 / mav.mass * np.array([fx, fy, fz]))

        # rotational kinematics
        quat_dot = 1 / 2 * np.array([[0, -p, -q, -r],
                                     [p, 0, r, -q],
                                     [q, -r, 0, p],
                                     [r, q, -p, 0]]) @ np.array([e0, e1, e2, e3])

        # rotational dynamics
        pqr_dot = np.array([
            mav.gamma1 * p * q - mav.gamma2 * q * r + mav.gamma3 * l + mav.gamma4 * n,
            mav.gamma5 * p * r - mav.gamma6 * (p ** 2 - r ** 2) + 1 / mav.Jy * m,
            mav.gamma7 * p * q - mav.gamma1 * q * r + mav.gamma4 * l + mav.gamma8 * n])

        # collect the derivative of the states
        x_dot = np.hstack([p_dot, uvw_dot, quat_dot, pqr_dot])
        x_dot = x_dot.flatten()
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6, 1))):
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame

        Vw_b = Quaternion2Rotation(self._state[6], self._state[7], self._state[8], self._state[9]) \
                @ np.array([wind[0],
                            wind[1],
                            wind[2]]) + np.array([wind[3],
                                                  wind[4],
                                                  wind[5]])

        Va_b = np.array([[self._state[3]],
                          [self._state[4]],
                          [self._state[5]]]) - Vw_b
        # compute airspeed (in body frame):
        self._Va = np.linalg.norm(Va_b)
        if self._Va == 0:
            self._alpha = 0
            self._beta = 0

        else:
            # compute angle of attack
            self._alpha = np.arctan2(Va_b.item(2), Va_b.item(0))
            self._alpha = self._alpha.item(0)

            # compute sideslip angle
            self._beta = np.arcsin(Va_b.item(1) / self._Va)

        # compute ground velocity
        Vg_v = Quaternion2Rotation(self._state[6], self._state[7], self._state[8], self._state[9]).T @ (Va_b + Vw_b)
        self._Vg = np.linalg.norm(Vg_v)

        self._chi = np.arctan2(Vg_v.item(1), Vg_v.item(0))
        self._gamma = np.arctan2(Vg_v[2], Vg_v.item(0))

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_t, delta_r)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mm, Mn)
        """

        # extract the states
        pn = self._state.item(0)
        pe = self._state.item(1)
        pd = self._state.item(2)
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        Va = self._Va
        # if Va == 0:
        #     Va = 0.000001
        alpha = self._alpha
        beta = self._beta

        delta_a = delta.item(0)
        delta_e = delta.item(1)
        delta_t = delta.item(2)
        delta_r = delta.item(3)

        phi, theta, psi = Quaternion2Euler(e0, e1, e2, e3)

        # gravity
        f_g = mav.mass * mav.gravity * np.array([[-np.sin(theta)],
                                                 [np.cos(theta) * np.sin(phi)],
                                                 [np.cos(theta) * np.cos(phi)]])
        # Motor Forces
        V_in = mav.V_max * delta_t
        # Quadratic formula to solve for motor speed
        a = mav.rho * np.power(mav.D_prop, 5) * mav.C_Q0 / ((2. * np.pi) ** 2)
        b = (mav.rho * np.power(mav.D_prop, 4) / (2. * np.pi)) ** 2 * mav.C_Q1 * Va + mav.KQ * mav.KQ / mav.R_motor
        c = mav.rho * np.power(mav.D_prop, 3) * mav.C_Q2 * Va ** 2 - (mav.KQ / mav.R_motor) * V_in + mav.KQ * mav.i0
        # Consider only positive root
        Omega_op = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2. * a)
        # add thrust and torque due to propeller
        T_p = (mav.rho * np.power(mav.D_prop, 4) * mav.C_T0 / (4 * np.pi ** 2)) * Omega_op ** 2 + (
                mav.rho * np.power(mav.D_prop, 3) * mav.C_T1 * Va / (2 * np.pi)) * Omega_op + (
                      mav.rho * mav.D_prop ** 2 * mav.C_T2 * Va ** 2)
        Q_p = (mav.rho * mav.D_prop ** 5 * mav.C_Q0 / (4 * np.pi ** 2)) * Omega_op ** 2 + (
                mav.rho * mav.D_prop ** 4 * mav.C_Q1 * Va / (2 * np.pi)) + (
                      mav.rho * mav.D_prop ** 3 * mav.C_Q2 * Va ** 2)

        self.thrust = T_p

        # Use the non-linear system to compute lift and drag
        sigma = (1 + np.exp(-mav.M * (alpha - mav.alpha0)) + np.exp(mav.M * (alpha + mav.alpha0))) \
                / ((1 + np.exp(-mav.M * (alpha - mav.alpha0))) * (
                1 + np.exp(mav.M * (alpha + mav.alpha0))))

        C_L = (1 - sigma) * (mav.C_L_0 + mav.C_L_alpha * alpha) + sigma * (
                2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))
        C_D = mav.C_D_p + (mav.C_L_0 + mav.C_L_alpha * alpha) ** 2 / (np.pi * mav.e * mav.AR)

        # 4.25
        C_chi = -C_D * np.cos(alpha) + C_L * np.sin(alpha)
        C_chi_q = -mav.C_D_q * np.cos(alpha) + mav.C_L_q * np.sin(alpha)
        C_chi_delta_e = -mav.C_D_delta_e * np.cos(alpha) + mav.C_L_delta_e * np.sin(alpha)
        C_Z = -C_D * np.sin(alpha) - C_L * np.cos(alpha)
        C_Z_q = -mav.C_D_q * np.sin(alpha) - mav.C_L_q * np.cos(alpha)
        C_Z_delta_e = -mav.C_D_delta_e * np.sin(alpha) - mav.C_L_delta_e * np.cos(alpha)

        Vs = np.sqrt(Va ** 2 + 2 * T_p / (mav.rho * np.pi / 4 * mav.D_prop ** 2))
        Vs = Va
        self._Vs = Vs

        Va = Vs # TODO 2nd Order LPF

        forces = f_g + np.array([[T_p], [0], [0]]) + 0.5 * mav.rho * Va ** 2 * mav.S_wing * \
                 np.array([[C_chi + C_chi_q * mav.c / (2 * Va) * q],
                           [mav.C_Y_0 + mav.C_Y_beta * beta + mav.C_Y_p * b / (2 * Va) * p + mav.C_Y_r * b / (2 * Va) * r],
                           [C_Z + C_Z_q * mav.c / (2 * Va) * q]]) + 0.5 * mav.rho * Va ** 2 * mav.S_wing * \
                 np.array([[C_chi_delta_e * delta_e],
                      [mav.C_Y_delta_a * delta_a + mav.C_Y_delta_r * delta_r],
                      [C_Z_delta_e * delta_e]])

        fx = forces.item(0)
        fy = forces.item(1)
        fz = forces.item(2)

        moments = 0.5 * mav.rho * Va ** 2 * mav.S_wing * \
                  np.array([[mav.b * (mav.C_ell_0 + mav.C_ell_beta * beta + mav.C_ell_p * b / (2 * Va) * p + mav.C_ell_r * b / (2 * Va) * r)],
                            [mav.c * (mav.C_m_0 + mav.C_m_alpha * alpha + mav.C_m_q * mav.c / (2 * Va) * q)],
                            [mav.b * (mav.C_n_0 + mav.C_n_beta * beta + mav.C_n_p * b / (2 * Va) * p + mav.C_n_r * b / (2 * Va) * r)]]) + \
                  0.5 * mav.rho * Va ** 2 * mav.S_wing * \
                  np.array([[mav.b * (mav.C_ell_delta_a * delta_a + mav.C_ell_delta_r * delta_r)],
                            [mav.c * (mav.C_m_delta_e * delta_e)],
                            [mav.b * (mav.C_n_delta_a * delta_a + mav.C_n_delta_r * delta_r)]]) + np.array([[Q_p], [0], [0]])

        l = moments.item(0)
        m = moments.item(1)
        n = moments.item(2)

        Mx = l
        My = m
        Mz = n

        forcesandmoments = np.array([[fx, fy, fz, Mx, My, Mz]])
        return forcesandmoments

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6], self._state[7], self._state[8], self._state[9])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.Vs = self._Vs
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = self._Vg
        self.msg_true_state.gamma = self._gamma
        self.msg_true_state.chi = self._chi
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
        self.msg_true_state.e0 = self._state[6]
        self.msg_true_state.e1 = self._state[7]
        self.msg_true_state.e2 = self._state[8]
        self.msg_true_state.e3 = self._state[9]
