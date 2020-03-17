"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import transfer_function
import parameters.aerosonde_parameters as p
from parameters.simulation_parameters import ts_simulation as Ts

def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    [e0, e1, e2, e3] = trim_state[6:10]
    [phi, theta, psi] = Quaternion2Euler(e0, e1, e2, e3)
    Va_star = mav._Va
    alpha_star = mav._alpha
    theta_star = theta
    chi_star = mav._chi
    delta_e_star = trim_input[1]
    delta_t_star = trim_input[2]

    #T_phi_delta_a
    a_phi_1 = -1/2*p.rho*mav._Va**2*p.S_wing*p.b*p.C_p_p*p.b/(2*mav._Va)
    a_phi_2 = 1/2*p.rho*mav._Va**2*p.S_wing*p.b*p.C_p_delta_a
    num = np.array([[a_phi_2]])
    den = np.array([[1, a_phi_1, 0]])
    T_phi_delta_a = transfer_function(num, den, Ts)

    #T_chi_phi
    num = np.array([[p.gravity/mav._Vg]])
    den = np.array([[1, 0]])
    T_chi_phi = transfer_function(num, den, Ts)

    #T_beta_delta_r
    a_beta_1 = -p.rho*mav._Va*p.S_wing/(2*p.mass)*p.C_Y_beta
    a_beta_2 = p.rho*mav._Va*p.S_wing/(2*p.mass)*p.C_Y_delta_r
    num = np.array([[a_beta_2]])
    den = np.array([[1, a_beta_1]])
    T_beta_delta_r = transfer_function(num, den, Ts)

    #T_theta_delta_e
    a_theta_1 = -p.rho*mav._Va**2*p.c*p.S_wing/(2*p.Jy)*p.C_m_q*p.c/(2*mav._Va)
    a_theta_2 = -p.rho*mav._Va**2*p.c*p.S_wing/(2*p.Jy)*p.C_m_alpha
    a_theta_3 = p.rho*mav._Va**2*p.c*p.S_wing/(2*p.Jy)*p.C_m_delta_e
    num = np.array([[a_theta_3]])
    den = np.array([[1, a_theta_1, a_theta_2]])
    T_theta_delta_e = transfer_function(num, den, Ts)

    #T_h_theta
    num = np.array([[mav._Vg]])
    den = np.array([[1, 0]])
    T_h_theta = transfer_function(num, den, Ts)

    #T_h_Va
    num = np.array([[theta]])
    den = np.array([[1, 0]])
    T_h_Va = transfer_function(num, den, Ts)

    #T_Va_delta_t
    a_v_1 = p.rho*Va_star*p.S_wing/p.mass*(p.C_D_0 + p.C_D_alpha*alpha_star + p.C_D_delta_e*delta_e_star) \
            + p.rho*p.S_prop/p.mass*p.C_prop*Va_star
    a_v_2 = p.rho*p.S_prop/p.mass*p.C_prop*p.k_motor**2*delta_t_star
    num = np.array([[a_v_2]])
    den = np.array([[1, a_v_1]])
    T_Va_delta_t = transfer_function(num, den, Ts)

    #T_Va_theta
    a_v_3 = p.gravity*np.cos(theta_star - chi_star)
    num = np.array([[-a_v_3]])
    den = np.array([[1, a_v_1]])
    T_Va_theta = transfer_function(num, den, Ts)

    # finite difference approach
    a_v_1 = p.rho*Va_star*p.S_wing/p.mass*(p.C_D_0 + p.C_D_alpha*alpha_star + p.C_D_delta_e*delta_e_star) \
            - 1.0/p.mass*dT_dVa(mav, mav._Va, delta_t_star)
    a_v_2 = 1.0/p.mass*dT_ddelta_t(mav, mav._Va, delta_t_star)
    a_v_3 = p.gravity * np.cos(theta - alpha_star)
    T_Va_delta_t = transfer_function(num=np.array([[a_v_2]]),
                                     den=np.array([[1, a_v_1]]),
                                     Ts=Ts)


    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input):


     return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
     return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    dx = 0.0001
    Va_old = mav._Va
    mav._Va = Va - dx
    mav._forces_moments(np.array([0.0, 0.0, delta_t, 0.0]))
    f_low = mav.thrust
    mav._Va = Va + dx
    mav._forces_moments(np.array([0.0, 0.0, delta_t, 0.0]))
    f_up = mav.thrust
    dThrust = (f_up-f_low)/(2*dx)
    mav._Va = Va_old
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    dx = 0.0001
    Va_old = mav._Va
    mav._Va = Va
    mav._forces_moments(np.array([0.0, 0.0, delta_t-dx, 0.0]))
    f_low = mav.thrust
    mav._forces_moments(np.array([0.0, 0.0, delta_t+dx, 0.0]))
    f_up = mav.thrust
    dThrust = (f_up-f_low)/(2*dx)
    mav._Va = Va_old
    return dThrust
