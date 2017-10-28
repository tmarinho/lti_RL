"""
Thiago Marinho
email: marinho@illinois.edu
Please feel free to use and modify this, but keep the above information. Thanks!
"""
from rlagent import RLagent
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
from controller import Controller
import trajGen
from model.lti import LTI
import numpy as np

control_frequency = 200 # Hz for attitude control loop
dt = 1.0 / control_frequency
time = [0.0]
SIMTIME = 5.0
t = np.arange(0.,SIMTIME+dt,dt)
reward = -9999
N = 100

def reward_fcn(states, des_states, F, M):
    x = states - des_states
    reward = - np.dot(x,x)
    #r = np.dot(M.T,M)[0,0]


def control(env, C, time):
    desired_state = 1
    u = C.run(env, desired_state)
    #reward_fcn(np.concatenate((quad.position(), quad.velocity()),axis=0),np.concatenate((desired_state.pos, desired_state.vel),axis=0),F,M)
    env.update(dt, u, time[0])
    time[0] += dt

def rl_control(env,agent,time):
    desired_state = 1
    s_t = env.state
    action = agent.do_action(env.state)
    env.update(dt, action, time[0])
    reward = env.reward(desired_state, action)
    agent.R.add(s_t, action, reward, env.state)

    minibatch = agent.R.sample(N)


    time[0] += dt

def main():
    #pdb.set_trace()
    pos = (0,0,0)
    attitude = (0,0,np.pi/2)
    env = LTI(0)
    C = Controller()
    agent = RLagent()
    # Simulation Loop
    while time[0] <= SIMTIME:
        rl_control(env,agent,time)

    plt.plot(t,env.hist)
    plt.show()


if __name__ == "__main__":
    main()
