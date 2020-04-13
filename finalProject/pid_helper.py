"""
pid_helper
"""
import sys
sys.path.append('..')

class pid_helper:
    def __init__(self, Ts=0.02, sigma=0.05):
        self.Ts = Ts
        self.integrator = 0.0
        self.error_delay_1 = 0.0
        self.error_dot_delay_1 = 0.0
        # gains for differentiator
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)

    def calc_derivative(self, error):
        error_dot = self.a1*self.error_dot_delay_1 + self.a2*(error - self.error_delay_1)
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot
        return error_dot

    def calc_integrator(self, error):
        self.integrator = self.integrator + (self.Ts/2)*(error + self.error_delay_1)
        self.error_delay_1 = error
        return self.integrator
