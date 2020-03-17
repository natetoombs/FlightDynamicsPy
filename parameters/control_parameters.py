import sys
sys.path.append('..')
import numpy as np
import chap5.transfer_function_coef as TF
import parameters.aerosonde_parameters as MAV
# # import chap4.mav_dynamics as mav
# from chap4.mav_dynamics import mav_dynamics
# import parameters.simulation_parameters as SIM
# # mav = mav_dynamics(SIM.ts_simulation)
g = MAV.gravity
# sigma =
Va0 = MAV.Va0
Va = Va0
Vg = Va
delta_a_max = .5
delta_e_max = .7
delta_r_max = .5
e_phi_max = np.radians(45)
e_theta_max = np.radians(35)
e_beta_max = np.radians(45)

# Longitudinal
wn_theta = np.sqrt(TF.a_theta2 + delta_e_max/e_theta_max*np.abs(TF.a_theta3))
zeta_theta = 0.4
W_h = 18
wn_h = 1/W_h*wn_theta
zeta_h = 0.707

# Lateral
wn_phi = np.sqrt(np.abs(TF.a_phi2)*delta_a_max/e_phi_max)
zeta_phi = 0.3
W_chi = 8
wn_chi = 1/W_chi*wn_phi
zeta_chi = 0.707
zeta_beta = 0.707

# Airspeed
wn_V = wn_h # FIX
zeta_V = 0.707

#----------roll loop-------------
roll_kp = delta_a_max/e_phi_max*np.sign(TF.a_phi2)
roll_kd = (2*zeta_phi*wn_phi - TF.a_phi1)/TF.a_phi2

#----------course loop-------------
course_kp = 2*zeta_chi*wn_chi*Vg/g
course_ki = wn_chi**2*Vg/g

#----------sideslip loop-------------
sideslip_kp = delta_r_max/e_beta_max*np.sign(TF.a_beta2)
sideslip_ki = 1/TF.a_beta2*((TF.a_beta1 + TF.a_beta2*sideslip_kp)/(2*zeta_beta))

# #----------yaw damper-------------
# yaw_damper_tau_r =
# yaw_damper_kp = 0.5

#----------pitch loop-------------
pitch_kp = delta_e_max/e_theta_max*np.sign(TF.a_theta3)
pitch_kd = (2*zeta_theta*wn_theta - TF.a_theta1)/TF.a_theta3
K_theta_DC = (pitch_kp*TF.a_theta3)/(TF.a_theta2 + pitch_kp*TF.a_theta3)

#----------altitude loop-------------
altitude_kp = (2*zeta_h*wn_h)/(K_theta_DC*Va)
altitude_ki = wn_h**2/(K_theta_DC*Va)
# altitude_zone = # FIX

#---------airspeed hold using throttle---------------
airspeed_throttle_kp = (2*zeta_V*wn_V - TF.a_V1)/TF.a_V2
airspeed_throttle_ki = wn_V**2/TF.a_V2
