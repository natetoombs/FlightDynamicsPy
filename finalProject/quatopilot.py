"""
 Quaternion-based Autopilot
    Nathan Toombs, 2020
"""
import sys
import numpy as np
sys.path.append('..')
from finalProject.pid_helper import pid_helper
from finalProject import quat_parameters as qp
from parameters import aerosonde_parameters as mav
from tools.tools import quaternion_multiply as qm
from tools.tools import quaternion_conjugate as qc

class quatopilot: #TODO: max deflections
    def __init__(self, ts_control):
        # initiate pid helpers
        self.ax_pid = pid_helper()
        self.ay_pid = pid_helper()
        self.az_pid = pid_helper()
        self.pn_pid = pid_helper()
        self.pe_pid = pid_helper()
        self.pd_pid = pid_helper()
        self.t_pid = pid_helper()
        self.hover_flag = False
        self.hover_pose = 0
        self.hover_pose_set = False
        self.L = 0
        self.M = 0
        self.N = 0

    def update(self, cmd, state): #need thrust or Vs,

        p_ref = cmd.p_ref
        q_ref = cmd.q_ref
        u_ref = cmd.u_ref

        ############# Straight & Off-Axis Flight #############
        psi_ref = np.arctan2(p_ref[1], p_ref[0])
        psi_array = np.array([np.cos(psi_ref), np.sin(psi_ref)])
        p_xy = np.array([state.pn, state.pe])
        p_refxy = np.dot(p_xy, psi_array)*psi_array # eq (36)

        p_ref = np.array([p_refxy[0], p_refxy[1], p_ref[2]])
        ###################################################

        # ################# Straight-up Flight ################
        # psi_ref = np.arctan2(p_ref[1], -p_ref[2])
        # psi_array = np.array([np.cos(psi_ref), np.sin(psi_ref)])
        # p_zy = np.array([-state.h, state.pe])
        # p_refzy = np.dot(p_zy, psi_array)*psi_array # eq (36)
        #
        # p_ref = np.array([p_ref[0], p_refzy[1], p_refzy[0]])
        # ###################################################

        # ########## Hovering (In development) ###############
        # if state.theta > np.radians(85):
        #     self.hover_flag = True
        # if self.hover_flag:
        #     if not self.hover_pose_set:
        #         self.hover_pose = np.array([state.pn, state.pe, -state.h])
        #         self.hover_pose_set = True
        #     p_ref = self.hover_pose
        # ###################################################

        q_des = self.position_controller(p_ref, q_ref, state)
        delta_a, delta_e, delta_r = self.attitude_controller(q_des, state)
        delta_t = self.thrust_controller(p_ref, u_ref, state)

        delta = np.array([delta_a, delta_e, delta_t, delta_r])
        return delta

    def position_controller(self, p_ref, q_ref, state):
        # Position Controller
        # Inputs: Reference position p_ref, position p from state (ned), q_ref
        # Outputs: q_des
        pn = state.pn
        pe = state.pe
        pd = -state.h

        p = np.array([pn, pe, pd])

        error_pi = p_ref - p  # Error in inertial position
        ep0 = self.pn_pid.calc_derivative(error_pi[0])
        ep1 = self.pe_pid.calc_derivative(error_pi[1])
        ep2 = self.pd_pid.calc_derivative(error_pi[2])
        error_pi_dot = np.array([ep0, ep1, ep2])

        qr0 = q_ref[0]
        qr1 = q_ref[1]
        qr2 = q_ref[2]
        qr3 = q_ref[3]

        qr02 = qr0 ** 2
        qr12 = qr1 ** 2
        qr22 = qr2 ** 2
        qr32 = qr3 ** 2

        # Rotation matrix from intertial to reference frame, eq (23)
        C_ri = np.array([[qr02 + qr12 - qr22 - qr32, 2 * (qr1 * qr2 + qr0 * qr3), 2 * (qr1 * qr3 - qr0 * qr2)],
                         [2 * (qr1 * qr2 - qr0 * qr3), qr02 - qr12 + qr22 - qr32, 2 * (qr2 * qr3 + qr0 * qr1)],
                         [2 * (qr1 * qr3 + qr0 * qr2), 2 * (qr2 * qr3 - qr0 * qr1), qr02 - qr12 - qr22 + qr32]])

        # Express errors in the reference frame, eq (22)
        [error_xp, error_yp, error_zp] = C_ri @ error_pi
        [error_xd, error_yd, error_zd] = C_ri @ error_pi_dot

        # If there is error in y, rotate about z to align thrust
        om_z = qp.k_pp * error_yp + qp.k_pd * error_yd  # Magnitude of rotation about z, eq (25)
        if np.abs(om_z) > np.radians(45):  # TODO comment here
            om_z = np.radians(45) * np.sign(om_z)
        qz = np.array([np.cos(om_z / 2), 0, 0, np.sin(om_z / 2)])  # Calculate quaternion rotation about z, eq (24)

        # If there is error in z, rotate about y to align thrust
        om_y = -(qp.k_pp * error_zp + qp.k_pd * error_zd)  # Magnitude of rotation about y, eq (27)
        if np.abs(om_y) > np.radians(45):  # TODO comment here
            om_y = np.radians(45) * np.sign(om_y)
        qy = np.array([np.cos(om_y / 2), 0, np.sin(om_y / 2), 0])  # Calculate quaternion rotation about y, eq (26)

        q_des = qm(qm(q_ref, qz), qy)  # Calculate q_des, eq (28)

        return q_des

    def attitude_controller(self, q_des, state):
        # Attitude Controller
        # Inputs: q from state quaternion, q_des from position controller, Vs or thrust for slipstream velocity
        # Outputs: delta (aer)

        q = np.array([state.e0, state.e1, state.e2, state.e3])  # Current Quaternion Orientation
        Vs = state.Vs
        if Vs == 0:
            Vs = 0.00001

        if np.linalg.norm(q + q_des) < np.linalg.norm(
                q - q_des):  # To ensure angular errors remain less than 180 deg, make q_ds negative
            q_des = -q_des  # (the negative of a quaternion is equal to the quaternion)
        q_star = qc(q)
        error_q = qm(q_star, q_des)  # Calculate the error quaternion, eq (2)
        # error_q = qm(q_des, np.conj(q))
        # error_q[1] = -error_q[1]
        Ex = 2 * np.arccos(error_q[0]) * error_q[1] / np.linalg.norm(
            error_q[1:])  # Orientation error about x axis, eq 8
        Ey = 2 * np.arccos(error_q[0]) * error_q[2] / np.linalg.norm(
            error_q[1:])  # Orientation error about y axis, eq 9
        Ez = 2 * np.arccos(error_q[0]) * error_q[3] / np.linalg.norm(
            error_q[1:])  # Orientation error about x axis, eq 10

        Ex_dot = self.ax_pid.calc_derivative(Ex)
        Ey_dot = self.ay_pid.calc_derivative(Ey)
        Ez_dot = self.az_pid.calc_derivative(Ez)
        L = (qp.k_ap * Ex + qp.k_ad * Ex_dot) * mav.Jx  # Desired moment about x, eq (11)
        M = (qp.k_ap * Ey + qp.k_ad * Ey_dot) * mav.Jy  # Desired moment about y, eq (12)
        N = (qp.k_ap * Ez + qp.k_ad * Ez_dot) * mav.Jz  # Desired moment about z, eq (13)
        self.L = L
        self.M = M
        self.N = N
        delta_a = L / (
                    1 / 2 * mav.rho * Vs ** 2 * mav.S_wing * mav.b * mav.C_ell_delta_a)  # Calculation for delta_a, eq (17)
        delta_e = M / (
                    1 / 2 * mav.rho * Vs ** 2 * mav.S_wing * mav.c * mav.C_m_delta_e)  # Calculation for delta_e, eq (18)
        delta_r = N / (
                    1 / 2 * mav.rho * Vs ** 2 * mav.S_wing * mav.b * mav.C_n_delta_r)  # Calculation for delta_r, eq (19)

        if np.abs(delta_a) > np.radians(45):
            delta_a = np.radians(45)*np.sign(delta_a)
        if np.abs(delta_e) > np.radians(45):
            delta_e = np.radians(45)*np.sign(delta_e)
        if np.abs(delta_r) > np.radians(45):
            delta_r = np.radians(45)*np.sign(delta_r)

        return delta_a, delta_e, delta_r

    def thrust_controller(self, p_ref, u_ref, state):
        # Thrust Controller
        # Inputs: p_ref (for height error), u_ref (for speed error), theta (from state)
        # Outputs: delta_t
        h_ref = -p_ref[2] # down position
        h = state.h
        error_h = h_ref - h

        u = state.Va
        error_u = u_ref - u

        theta = state.theta
        alpha = state.alpha

        u_dot = qp.k_up * error_u + (qp.k_hp * error_h + qp.k_hi * self.t_pid.calc_integrator(error_h)) * np.sin(
            theta)  # Calculate the needed acceleration in x, eq (29)

        sigma = (1 + np.exp(-mav.M * (alpha - mav.alpha0)) + np.exp(mav.M * (alpha + mav.alpha0))) \
                / ((1 + np.exp(-mav.M * (alpha - mav.alpha0))) * (
                1 + np.exp(mav.M * (alpha + mav.alpha0))))

        C_L = (1 - sigma) * (mav.C_L_0 + mav.C_L_alpha * alpha) + sigma * (
                2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)) # UAV Book eq. 4.9
        C_D = mav.C_D_p + (mav.C_L_0 + mav.C_L_alpha * alpha) ** 2 / (np.pi * mav.e * mav.AR) # UAV Book eq. 4.11

        L = 1/2*mav.rho*u**2*mav.S_wing*C_L
        D = 1/2*mav.rho*u**2*mav.S_wing*C_D

        T = D * np.cos(theta) + mav.mass * mav.gravity * np.sin(theta) - L * np.sin(theta) + mav.mass * u_dot
        if T < 0:
            T = 0
        # TODO: Add equations 32, 33 for slipstream

        # Vs_d_e = np.sqrt(self.M /(.5*mav.rho*mav.S_wing*mav.c*mav.C_m_delta_e*np.radians(45)))
        # Vs_d_a = np.sqrt(self.L / (.5 * mav.rho * mav.S_wing * mav.b * mav.C_ell_delta_a * np.radians(45)))
        # Vs_d_r = np.sqrt(self.N / (.5 * mav.rho * mav.S_wing * mav.b * mav.C_n_delta_r * np.radians(45)))
        # Vs_d = max(Vs_d_e, Vs_d_a, Vs_d_r)
        # T_slip = mav.rho*np.pi/4*mav.D_prop**2/2*(Vs_d**2 - state.Va**2) # eq 33
        # T_slip = max(T_slip, 20)
        # if Vs_d > state.Va:
        #     T = T + T_slip
        # print(T)
        delta_t = T / 54  # Using terrible linear model TODO: Make a better one
        if delta_t > 6:
            delta_t = 6*np.sign(delta_t) # Limit the throttle voltage to 6 times the max (arbitrary)
        elif delta_t < 1:
            delta_t = 0

        return delta_t
