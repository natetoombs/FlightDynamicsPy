import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(60) # approach angle for large distance from straight-line path
        self.k_path = 0.01 # proportional gain for straight-line path following
        self.k_orbit = 0.4 # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        r = path.line_origin
        rn = r[0]
        re = r[1]
        rd = r[2]

        q = path.line_direction
        qn = q[0]
        qe = q[1]
        qd = q[2]

        pn = state.pn
        pe = state.pe
        h = state.h
        p = np.array([pn, pe, -h])
        chi = state.chi

        chi_q = self._wrap(np.arctan2(qe, qn), chi) # 10.1
        R_ip = np.array([[np.cos(chi_q), np.sin(chi_q), 0],
                         [-np.sin(chi_q), np.cos(chi_q), 0],
                         [0, 0, 1]])
        e_pi = p - r
        e_p = R_ip @ e_pi
        k = np.array([0, 0, 1])
        n = np.cross(q, k) / np.linalg.norm(np.cross(q, k))
        s = e_pi - np.dot(e_pi, n)*n
        h_c = -rd + np.sqrt(s[0]**2 + s[1]**2) * (qd/(np.sqrt(qn**2 + qe**2))) # Supplement 10.5

        e_py = e_p[1]
        e_py = -np.sin(chi_q)*(pn - rn) + np.cos(chi_q)*(pe - re)
        chi_c = chi_q -self.chi_inf*2/np.pi*np.arctan(self.k_path*e_py) # 10.8

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = 0.0

    def _follow_orbit(self, path, state):

        pn = state.pn
        pe = state.pe
        pd = -state.h
        g = 9.81
        c = path.orbit_center
        cn = c[0]
        ce = c[1]
        cd = c[2]
        rho = path.orbit_radius
        chi = state.chi

        if path.orbit_direction == 'CW':
            lam = 1
        else:
            lam = -1

        h_c = -cd

        d = np.sqrt((pn - cn)**2 + (pe - ce)**2)

        varphi = np.arctan2(pe-ce, pn-cn)
        while varphi - chi < -np.pi:
            varphi += 2*np.pi
        while varphi - chi > np.pi:
            varphi -= 2*np.pi

        chi_c = varphi + lam*(np.pi/2 + np.arctan(self.k_orbit*(d - rho)/rho))

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = np.radians(0)

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

