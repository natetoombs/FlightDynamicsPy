"""
tools that dr. beard didn't upload
"""

import numpy as np
import math

def quaternion_multiply(quaternion0, quaternion1):
    a, b, c, d = quaternion0
    e, f, g, h = quaternion1
    return np.array([a*e - b*f - c*g - d*h,
                        b*e + a*f + c*h - d*g,
                        a*g - b*h + c*e + d*f,
                        a*h + b*g - c*f + d*e])

def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def Euler2Quaternion(phi, theta, psi):
    e0 = math.cos(psi/2)*math.cos(theta/2)*math.cos(phi/2) + math.sin(psi/2)*math.sin(theta/2)*math.sin(phi/2)
    e1 = math.cos(psi/2)*math.cos(theta/2)*math.sin(phi/2) - math.sin(psi/2)*math.sin(theta/2)*math.cos(phi/2)
    e2 = math.cos(psi/2)*math.sin(theta/2)*math.cos(phi/2) + math.sin(psi/2)*math.cos(theta/2)*math.sin(phi/2)
    e3 = math.sin(psi/2)*math.cos(theta/2)*math.cos(phi/2) - math.cos(psi/2)*math.sin(theta/2)*math.sin(phi/2)

    return [e0, e1, e2, e3]

def Quaternion2Euler(e0, e1, e2, e3):

    phi = np.arctan2(2*(e0*e1+e2*e3), (e0**2+e3**2-e1**2-e2**2))
    theta = np.arcsin(2*(e0*e2-e1*e3))
    psi = np.arctan2(2*(e0*e3+e1*e2), (e0**2+e1**2-e2**2-e3**2))

    return [phi, theta, psi]

def Quaternion2Rotation(e0, e1, e2, e3):
    phi, theta, psi = Quaternion2Euler(e0, e1, e2, e3)
    c_theta = np.cos(theta)
    c_phi = np.cos(phi)
    c_psi = np.cos(psi)
    s_theta = np.sin(theta)
    s_phi = np.sin(phi)
    s_psi = np.sin(psi)

    R_v_b = np.array([[c_theta*c_psi, c_theta*s_psi, -s_theta],
                      [s_phi*s_theta*c_psi-c_phi*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, s_phi*c_theta],
                      [c_phi*s_theta*c_psi+s_phi*s_psi, c_phi*s_theta*s_psi-s_phi*c_psi, c_phi*c_theta]])

    return R_v_b

def Euler2Rotation(phi, theta, psi):
    c_theta = np.cos(theta)
    c_phi = np.cos(phi)
    c_psi = np.cos(psi)
    s_theta = np.sin(theta)
    s_phi = np.sin(phi)
    s_psi = np.sin(psi)

    R_v_b = np.array([[c_theta*c_psi, c_theta*s_psi, -s_theta],
                      [s_phi*s_theta*c_psi-c_phi*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, s_phi*c_theta],
                      [c_phi*s_theta*c_psi+s_phi*s_psi, c_phi*s_theta*s_psi-s_phi*c_psi, c_phi*c_theta]])

    return R_v_b
