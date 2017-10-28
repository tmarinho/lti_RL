from rlagent import RLagent
import tensorflow as tf
import numpy as np

N = 10
agent = RLagent()
agent.R.add(1, 2, 0, 1.5)
agent.R.add(1, 2.5, 0, 1.5)
n = agent.R.size()
if n >= N:
    n = N

minibatch = np.asarray(agent.R.sample(n))
#print agent.sess.run(agent.mu_NN.params)
#print agent.sess.run(agent.target_mu_NN.params)
#print agent.do_action([[3]])

s_old_s = minibatch[:,0]
action_s = minibatch[:,1]
reward_s = minibatch[:,2]
s_new_s = np.expand_dims(minibatch[:,3],axis = 1)

print [s_new_s.T]

print agent.do_action_target(s_new_s)
