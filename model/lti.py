import numpy as np
import scipy.integrate as integrate
from math import sin, cos
class LTI:
    def __init__(self, state_0):
        """ pos = [x,y,z] attitude = [rool,pitch,yaw]
            """
        self.state = state_0
        self.hist = []
    def state_dot(self, state, t, u, time):
        x1 = state
        b = 3
        a = 2
        state_dot = 0.0
        d =0.14*sin(time*5)
        print d
        state_dot  = -a*x1 + b*u +d

        return state_dot

    def reward(self,desired_state,u):
        return -(self.state - desired_state)^2 - 0.2*u^2

    def update(self, dt,u, time):
        #saturate u
        u = np.clip(u,-10,10)
        out = integrate.odeint(self.state_dot, self.state, [0,dt], args = (u,time))
        #print out
        self.state = out[1]
        self.hist.append(np.array(self.state[0]))
