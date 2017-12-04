# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model:

    def __init__(self, data, label, drop_out_prob,drop_out_prob2, config):
        self.data = data
        self.label = label
        self.time_steps = config['n_timesteps']
        self.n_input = config['n_inputs']
        self.n_classes = config['n_classes']
        self.drop_out_prob = drop_out_prob
        self.drop_out_prob2 = drop_out_prob2
        self.hidden_size = config['hidden_size']
        self.hidden_size_smaller = config['hidden_size_small']
        self.prediction
        self.optimize
        self.loss

    def build_mlp(self,input_data):
        network = input_data
        network = tf.nn.dropout(network, self.drop_out_prob2)
        network = tf.contrib.layers.fully_connected(network,self.hidden_size_smaller,scope = 'fc0')
        network = tf.nn.dropout(network, self.drop_out_prob2)
        network = tf.contrib.layers.fully_connected(network,self.hidden_size,scope = 'fc1')
        network = tf.nn.dropout(network, self.drop_out_prob2)
        network = tf.contrib.layers.fully_connected(network,self.n_classes,activation_fn=None,scope = 'fc2')
        return network

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        drop_input = tf.nn.dropout(self.data, self.drop_out_prob)
        formatted_input = tf.unstack(drop_input,self.time_steps, 1)
        lstm_layer=rnn.BasicLSTMCell(self.n_input,forget_bias=1)
        lstm_layer2 = rnn.BasicLSTMCell(self.n_input,forget_bias=1)
        cell = rnn.MultiRNNCell([lstm_layer,lstm_layer2])
        outputs , state = tf.nn.static_rnn(cell,formatted_input,dtype='float32')
        mlp = self.build_mlp(outputs[-1])
        return mlp

    @define_scope
    def loss(self):
        loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.label,axis=-1),tf.nn.l2_normalize(self.prediction,axis=-1),axis=-1)
        return loss
    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(self.loss)