from rlagent import RLagent
import tensorflow as tf
import numpy as np

N = 10
agent = RLagent()
agent.R.add(1.0, 2.0, 0.0, 1.5)
agent.R.add(1.0, 33.0, 1.0, 3.0)
agent.R.add(1.0, 33.0, 2.0, 3.0)


n = agent.R.size()
if n >= N:
    n = N

minibatch = np.asarray(agent.R.sample(n))
#print agent.sess.run(agent.mu_NN.params)
#print agent.sess.run(agent.target_mu_NN.params)
#print agent.do_action([[3]])

s_old_s =np.expand_dims(minibatch[:,0],axis = 1)
action_s = np.expand_dims(minibatch[:,1],axis = 1)
reward_s = np.expand_dims(minibatch[:,2],axis = 1)
#reward_s = minibatch[:,2]
s_new_s = np.expand_dims(minibatch[:,3],axis = 1)

#print agent.s_next
#print agent.target_mu_NN.y_output

#print tf.concat([agent.s_next,agent.target_mu_NN.y_output],1)

print agent.update_critic(s_old_s,action_s,s_new_s,reward_s)
