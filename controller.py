import numpy as np
from math import sin, cos

# PD Controller for quacopter
# Return [F, M] F is total force thrust, M is 3x1 moment matrix

# Constants
class Controller:
    def __init__(self):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        self.integral = 0
        self.kp = 5
        self.ki = 0.1

    def run(self, lti, des_state):
        x = lti.state
        e = des_state - x
        self.integral =  self.integral + e
        u = self.kp  *e + self.ki* self.integral


        return u
