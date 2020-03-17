"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
from tools.transfer_function import transfer_function
from tools.tools import Quaternion2Euler
from tools.wrap import wrap
from chap6.pid_control import pid_control #, pi_control, pd_control_with_rate
from message_types.msg_state import msg_state


class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pid_control( #pd_control_with_rate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = pid_control( #pi_control(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(35))
        self.sideslip_from_rudder = pid_control( #pi_control(
                        kp=AP.sideslip_kp,
                        ki=AP.sideslip_ki,
                        Ts=ts_control,
                        limit=np.radians(45))
        # self.yaw_damper = transfer_function( # FIX
        #                 num=np.array([[AP.yaw_damper_kp, 0]]),
        #                 den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
        #                 Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pid_control( #pd_control_with_rate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = pid_control( #pi_control(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = pid_control( #pi_control(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = msg_state()

    def update(self, cmd, state):

        Va_c = cmd.airspeed_command
        chi_c = cmd.course_command
        h_c = cmd.altitude_command

        phi = state.phi
        theta = state.theta
        psi = state.psi
        # e0 = state._state.item(6) # FIX
        # e1 = state._state.item(7)
        # e2 = state._state.item(8)
        # e3 = state._state.item(9)
        # phi, theta, psi = Quaternion2Euler(e0, e1, e2, e3)

        h = state.h
        p = state.p
        q = state.q
        r = state.r

        # wrap the commanded course
        chi_c = wrap(chi_c, state.chi)

        # lateral autopilot
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        delta_a = self.roll_from_aileron.update_with_rate(phi_c, phi, p)
        delta_r = self.sideslip_from_rudder.update(0, state.beta)

        # longitudinal autopilot
        theta_c = self.altitude_from_pitch.update(h_c, h)
        delta_e = self.pitch_from_elevator.update_with_rate(theta_c, theta, q)
        delta_t = self.airspeed_from_throttle.update(Va_c, state.Va)
        if delta_t < 0:
            delta_t = 0

        # construct output and commanded states
        delta = np.array([[delta_a], [delta_e], [delta_t], [delta_r]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
