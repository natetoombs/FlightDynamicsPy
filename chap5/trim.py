"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion

def compute_trim(mav, Va, gamma):
    # define initial state and input
    state0 = np.zeros(13)
    state0[6] = 1
    state0[2] = 0
    state0[3] = Va * np.cos(gamma)
    state0[5] = -Va * np.sin(gamma)

    delta0 = np.array([0, 0, .5, 0])
    x0 = np.concatenate((state0, delta0), axis=0)

    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[:13]]).T
    trim_state = trim_state.flatten()
    trim_input = np.array([res.x[13:]]).T
    trim_input = trim_input.flatten()
    return trim_state, trim_input

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):
    state = x[:13]
    delta = x[13:]

    mav._state = x[:13]

    # mav._state = mav._state.flatten()
    mav._update_velocity_data()
    x_dot = mav._derivatives(state, delta)

    R = 3

    x_dot_star = np.zeros(13)
    x_dot_star[2] = -Va * np.sin(gamma)
    x_dot_star[8] = Va / R * np.cos(gamma)

    diff = x_dot - x_dot_star
    diff[0:2] = np.zeros(2)
    J = np.linalg.norm(diff)

    return J

