import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        # dubins path parameters
        self.dubins_path = dubins_parameters()
        self.new_waypoint_path = True
        self.flag_path_changed = False

    def update(self, waypoints, radius, state):
        def update(self, waypoints, radius, state):
            # this flag is set for one time step to signal a redraw in the viewer
            if self.path.flag_path_changed == True:
                self.path.flag_path_changed = False
            if waypoints.num_waypoints == 0:
                waypoints.flag_manager_requests_waypoints = True
            else:
                if waypoints.type == 'straight_line':
                    self.line_manager(waypoints, state)
                elif waypoints.type == 'fillet':
                    self.fillet_manager(waypoints, radius, state)
                elif waypoints.type == 'dubins':
                    self.dubins_manager(waypoints, radius, state)
                else:
                    print('Error in Path Manager: Undefined waypoint type.')
            return self.path

    def line_manager(self, waypoints, state):
        w = waypoints.ned
        w_idx = 1 # Indexed at 0
        sm = 1 #state machine


    def fillet_manager(self, waypoints, radius, state):
        w = waypoints.ned
        p = np.array([state.pn, state.pe, -state.h])
        N = waypoints.num_waypoints

        assert (N >= 3), "Need more waypoints!"

        if self.new_waypoint_path:
            self.initialize_pointers()

        i = self.ptr_current
        self.manager_state = 1

        q_prev = (w[i] - w[i - 1])/np.linalg.norm(w[i] - w[i - 1])
        q = (w[i + 1] - w[i])/np.linalg.norm(w[i + 1] - w[i])
        vartheta = np.arccos(-q_prev.T @ q)

        if self.manager_state == 1:
            flag = 1
            r = w[i-1] #TODO: Check this (w[i])
            q = q_prev
            z = w[i] - (radius/np.tan(vartheta/2))*q_prev

            self.halfspace_r = np.array([z])
            self.halfspace_n = np.array([q_prev])

            # Check if in half space &
            # Tell waypoint viewer to replot the path
            if self.flag_path_changed:
                self.path.flag_path_changed = True
                self.flag_path_changed = False
            if self.inHalfSpace(p):
                self.manager_state = 2
                self.flag_path_changed = True
        elif self.manager_state == 2:
            flag = 2
            c = w[i] + (radius/np.sin(vartheta/2))*(q_prev-q)/np.linalg.norm(q_prev-q)
            rho = radius
            lam = np.sign(q_prev[0]*q[1] - q_prev[1]*q[0])
            z = w[i] + (radius/np.tan(vartheta/2))*q
            if self.inHalfSpace(p):
                self.increment_pointers(waypoints.num_waypoints)
                self.manager_state = 1
                self.flag_path_changed = True

    def dubins_manager(self, waypoints, radius, state):

    def initialize_pointers(self):
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2

        self.new_waypoint_path = False

    def increment_pointers(self, num_waypoints):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        self.ptr_next = self.ptr_next + 1

        if self.ptr_next == num_waypoints:
            self.ptr_next = 0

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

