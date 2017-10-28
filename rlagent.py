import numpy as np
from math import sin, cos
from RLlib import NN
import tensorflow as tf
from collections import deque
import random
#from copy import deepcopy

MAX_REPLAY_SIZE = 10000
DISCOUNTFACTOR = 0.99
CONTROL_DIM = 1
STATE_DIM = 1

class RLagent:
    def __init__(self):

        ### Initialize Neural Network
        # Critic Network
        self.Q_out              = tf.placeholder(tf.float32, [None, 1])
        self.s_and_a            = tf.placeholder(tf.float32, [None, CONTROL_DIM+STATE_DIM])
        self.Q_NN               = NN(self.s_and_a, self.Q_out,[16,16],CONTROL_DIM+STATE_DIM,1)
        self.target_Q_NN        = NN(self.s_and_a, self.Q_out,[16,16],CONTROL_DIM+STATE_DIM,1)
        # Actor Network Output
        self.s                  = tf.placeholder(tf.float32, [None, STATE_DIM])
        self.mu_out             = tf.placeholder(tf.float32, [None, CONTROL_DIM])
        self.mu_NN              = NN(self.s, self.mu_out,[1,1],STATE_DIM,CONTROL_DIM)
        self.target_mu_NN       = NN(self.s, self.mu_out,[1,1],STATE_DIM,CONTROL_DIM)

        #var_copy = []
        #for var, var_target in zip(self.mu_NN.params,self.target_mu_NN.params):
        #    var_copy.append(var_target.assign(var))
        #copy_op = tf.group(*var_copy)

        copy_op = self.update_target_network(1,self.mu_NN,self.target_mu_NN)
        #Replay Buffer
        self.R                  = Replay(MAX_REPLAY_SIZE)
        ### Initialize the TF Session
        init = tf.global_variables_initializer()  ### Initialize the variables, Parameters
        #TODO Use context manager to manage session
        self.sess = tf.Session()                       ### Start TF Session
        self.sess.run(init)
        self.sess.run(copy_op)

    def update_target_network(self,tau,N,N_target):
        var_copy = []
        for var, var_target in zip(N.params,N_target.params):
            var_copy.append(var_target.assign(tau*var+(1-tau)*var_target))
        return tf.group(*var_copy)

    def do_action(self,state):
        return self.sess.run(self.mu_NN.y_output, {self.s :state})
    
    def do_action_target(self,state):
        return self.sess.run(self.target_mu_NN.y_output, {self.s :state})
#    def run(self, lti, des_state):

#        return



class Replay:

    def __init__(self, max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self, observation, action, reward, observation2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((observation, action, reward, observation2))

    def sample(self, count):
        return random.sample(self.transitions, count)

    def size(self):
        return len(self.transitions)
