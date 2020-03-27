"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.tools import Euler2Rotation
from tools.wrap import wrap
from message_types.msg_state import msg_state

import numpy as np
from scipy import stats

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.abs_pressure)/(MAV.gravity*MAV.rho)
        self.estimated_state.Va = np.sqrt(2/MAV.rho*self.lpf_diff.update(np.abs(measurements.diff_pressure)))

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha*self.y + (1 - self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        self.Q = np.eye(2) * 0.1
        self.R_gyro = np.eye(3) * SENSOR.gyro_sigma ** 2
        self.R_accel = np.eye(3) * SENSOR.accel_sigma**2
        self.N = 15   # number of prediction step per sample
        self.xhat = np.array([MAV.phi0, MAV.theta0]) # initial state: phi, theta
        self.P = np.eye(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        p = state.p
        q = state.q
        r = state.r
        phi = x[0]
        theta = x[1]

        _f = np.array([p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta),
                       q*np.cos(phi) - r*np.sin(phi)])
        return _f

    def h(self, x, state):
        # measurement model y
        p = state.p
        q = state.q
        r = state.r
        phi = x[0]
        theta = x[1]
        Va = state.Va

        _h = np.array([q*Va*np.sin(theta) + MAV.gravity*np.sin(theta),
                       r*Va*np.cos(theta) - p*Va*np.sin(theta) - MAV.gravity*np.cos(theta)*np.sin(phi),
                       -q*Va*np.cos(theta) - MAV.gravity*np.cos(theta)*np.cos(phi)])
        return _h

    def propagate_model(self, state):
        # model propagation
        phi = state.phi
        theta = state.theta

        for i in range(0, self.N):
             # propagate model
            self.xhat = self.xhat + (self.Ts)*self.f(self.xhat, state) # Was self.Ts/N, helped without N
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise # TODO WHAT IS THIS
            G = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],    # Slide 34
                          [0, np.cos(phi), -np.sin(phi)]])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(2) + A*self.Ts + A @ A*self.Ts**2/2
            # G_d =
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2*(self.Q + G @ self.R_gyro @ G.T) # What is going on here?
            # Why was there Q and also R_gyro?
    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z]).T
        S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        if stats.chi2.sf((y - h).T @ S_inv @ (y - h), df=3)>0.01:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(2) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_accel @ L.T
            self.xhat = self.xhat + L @ (y - h)

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi
    def __init__(self):
        self.Q = np.eye(7) * 1
        self.Q[4, 4] = 1
        self.Q[5, 5] = 1
        self.Q[6, 6] = 1
        self.R = np.diag([SENSOR.gps_n_sigma**2, SENSOR.gps_e_sigma**2, SENSOR.gps_Vg_sigma**2, SENSOR.gps_course_sigma**2])
        self.R_pseudo = np.diag([0.01, 0.01]) # Covariance for pseudo sensors
        self.N = 5  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([MAV.pn0, MAV.pe0, MAV.Va0, 0.0, 0.0, 0.0, MAV.psi0])
        self.P = np.eye(7)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        q = state.q
        r = state.r
        phi = state.phi
        theta = state.theta
        psi = x[6]
        Va = state.Va
        Vg = x[2]
        g = MAV.gravity
        chi = x[3]
        wn = x[4]
        we = x[5]
        psi_dot = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

        _f = np.array([Vg*np.cos(chi), #is psi_dot the same as r?
                       Vg*np.sin(chi),
                       ((Va*np.cos(chi)+wn)*(-Va*psi_dot*np.sin(psi)) + (Va*np.sin(psi)+we)*(Va*psi_dot*np.cos(psi)))/Vg,
                       g/Vg*np.tan(phi),
                       0,
                       0,
                       psi_dot])
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        Vg = x[3]
        pn = x[0]
        pe = x[1]
        chi = x[3]

        _h = np.array([pn, pe, Vg, chi])
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangle pseudo measurement
        psi = x[6]
        Va = state.Va
        Vg = x[2]
        chi = x[3]
        wn = x[4]
        we = x[5]

        _h = np.array([Va*np.cos(psi) + wn - Vg*np.cos(chi),
                       Va*np.sin(psi) + we - Vg*np.sin(chi)])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat = self.xhat + (self.Ts/self.N)*self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(7) + A*self.Ts + A @ A*self.Ts**2/2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2*self.Q

    def measurement_update(self, state, measurement): #TODO NEXT
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([0, 0])
        Ci = C[:, 4:6]
        Li = self.P[4:6, 4:6] @ Ci.T @ np.linalg.inv(self.R_pseudo + Ci @ self.P[4:6, 4:6] @ Ci.T)
        self.P[4:6, 4:6] = (np.eye(2) - Li @ Ci) @ self.P[4:6, 4:6] @ (np.eye(2) - Li @ Ci).T + Li@self.R_pseudo@Li.T # Supplement page 145
        self.xhat[4:6] = self.xhat[4:6] + Li @ (y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course])
            Ci = C[:, 0:4]
            L = self.P[0:4, 0:4] @ Ci.T @ np.linalg.pinv(self.R + Ci @ self.P[0:4, 0:4] @ Ci.T) #used pinv to prevent singular matrix error
            self.P[0:4, 0:4] = (np.eye(4) - L @ Ci) @ self.P[0:4, 0:4] @ (np.eye(4) - L @ Ci).T + L@self.R@L.T # Supplement page 145
            self.xhat[0:4] = self.xhat[0:4] + L @ (y - h)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, state):
    # compute jacobian of fun with respect to x
    f = fun(x, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df
    return J
