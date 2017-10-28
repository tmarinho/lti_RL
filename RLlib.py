import tensorflow as tf
import numpy as np
import timeit

learning_rate = 0.00001


class Hidden_layer:
    def __init__(self, x_input, n_in, n_out, activation='sigmoid'):
        self.W = tf.Variable(tf.random_normal([n_in, n_out]))  ### initialize the W with Normal distribution
        self.b = tf.Variable(tf.zeros([n_out]))                ###                b with zeros
        self.x = x_input

        if activation == 'sigmoid':
            self.y = tf.nn.sigmoid(tf.matmul(self.x, self.W)+self.b)

        self.params = [self.W, self.b]

class Output_layer:
    def __init__(self, x_input, n_in, n_out, activation='linear'):
        self.W = tf.Variable(tf.random_normal([n_in, n_out]))
        self.b = tf.Variable(tf.zeros([n_out]))
        self.x = x_input
        if activation == 'linear':
            self.y = tf.matmul(self.x, self.W)+self.b

        self.params = [self.W, self.b]
class NN:
    def __init__(self, x_input, y_target, n_layer_struct, n_in, n_out):
        ### ex. n_layers = [20,10] means that the ouput of two hidden_layers are 20 in the 1st level, and 10 in the 2nd.
        self.n_layer_struct = n_layer_struct
        self.n_level = len(self.n_layer_struct)

        ### Stacking and  Connecting Hidden Layers
        self.hidden_layers = []

        if self.n_level == 1:
            self.hidden_layers.append(
                 Hidden_layer(x_input=x_input, n_in=n_in, n_out=n_layer_struct[0])
            )

        else:
            for i in range(self.n_level):
                if i==0:
                    self.hidden_layers.append(
                         Hidden_layer(x_input=x_input, n_in=n_in, n_out=n_layer_struct[i])
                    )
                else:
                    self.hidden_layers.append(
                         Hidden_layer(x_input=self.hidden_layers[i-1].y, n_in=n_layer_struct[i-1], n_out=n_layer_struct[i])
                    )
        ### Output Layer
        if self.n_level == 1:
            self.output_layer = Output_layer(x_input = self.hidden_layers[0].y, n_in = n_layer_struct[0], n_out=n_out)
        else:
            self.output_layer = Output_layer(x_input = self.hidden_layers[i].y, n_in = n_layer_struct[i], n_out=n_out)

        ### Parameters
        self.params=[]
        if self.n_level==1:
            self.params+=self.hidden_layers[0].params
        else:
            for i in range(self.n_level):
                self.params+=self.hidden_layers[i].params
        self.params+=self.output_layer.params

        ### Input and Output of MLP
        self.x_input  = x_input
        self.y_output = self.output_layer.y

        ### Mean Squares Error (MSE)
        self.mse = tf.reduce_mean((y_target - self.y_output)**2)

        ### Gradient of MSE respect to Params given Data
        self.grads = tf.gradients(self.mse, self.params)

        var_updates = []

        for grad, var in zip(self.grads, self.params):
            var_updates.append(var.assign_sub(learning_rate * grad))
        self.train_op = tf.group(*var_updates)
