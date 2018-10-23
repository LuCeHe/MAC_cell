#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:13:12 2018

@author: hey
"""

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras import backend as K
from keras.engine.topology import Layer

from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

import numpy as np



# MAC cell
# https://arxiv.org/pdf/1803.03067.pdf
class ControlUnit(Layer):
  
  
    def __init__(self, d_dim, **kwargs):
        self.output_dim = output_dim
        self.d_dim = d_dim
        self.num_hops = num_hops
        super(ControlUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][2]
    
        initial_W_d2d_value = np.random.uniform(0, 1, size=[self.d_dim, 2*self.d_dim])
        initial_b_d_value = np.random.uniform(0, 1, size=[self.d_dim])
    
        initial_W_1d_value = np.random.uniform(0, 1, size=[1, self.d_dim])
        initial_b_1_value = np.random.uniform(0, 1, size=[1])

        self.input_dim = input_shape[0][1]
        self.W_d2d = K.variable(initial_W_d2d_value)
        self.b_d   = K.variable(initial_b_d_value)
    
        self.W_1d  = K.variable(initial_W_1d_value)
        self.b_1   = K.variable(initial_b_1_value)

        self.trainable_weights = [self.W_d2d, self.b_d, self.W_1d, self.b_1]
        super(MemoryRepresentation, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i_1 = inputs[0]
        q = inputs[1]
        cw_s = inputs[2]
    
        # equation c1  
        conc_cq = K.concatenate([c_i_1, q], axis=1)
    
        cq_i = K.dot(conc_cq, K.transpose(self.W_d2d))
        cq_i = K.bias_add(cq_i, self.b_d, data_format=None)  
    
        # equation c2.1  
        cqcw = cq_i*cw_s
    
        ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
    
        # equation c2.2
        cv_is = K.softmax(ca_is)
    
        # equation c2.3
        c_i = cv_is*cw_s
    
        return c_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  

class writeUnit(Layer):
    pass
class readUnit(Layer):
    pass
class MAC_cell(Layer):
    pass
# MAC cell with memory to the controller
class mcMAC_cell(Layer):
    pass
  
  
  
# recurrent MAC
class RecurrentMAC_cell(Layer):
    pass


# adaptive computation time
# https://arxiv.org/abs/1603.08983

class ACT_RMAC_cell(Layer):
    pass


def test_ControlUnit():
    d = 2
    
    
    question_1 = np.array([1, 2, 5, 2, 4])
    n_timesteps = len(question_1)
    question_1 = question_1.reshape(1, n_timesteps, 1)
    question_2 = np.array([1, 2, 5, 2])
    n_timesteps = len(question_2)
    question_2 = question_2.reshape(1, n_timesteps, 1)
  
    
    for question in [question_2]:
      
        # biLSTM
      
        inputs = Input(shape=(None,1), name='question')
        #biLSTM_Model = Sequential()
        output = Bidirectional(LSTM(d, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=None)(inputs)
        
      
        biLSTM_Model = Model(inputs = inputs, output = output)
        cw = biLSTM_Model.predict(question)
      
        for cwi in cw:
            print(cwi.shape)
            print('')  
            print(cwi)
            print('') 
            print('')  
    
      
      
'''
  # data to train
  
  c_i_1 = np.random.uniform(0, 1, size=[d])
  
  
  # build model  
  
  
  c_input = Input(shape=(story_maxlen, len(vocab)), name='c_input')
  q_input = Input(shape=(story_maxlen, len(vocab)), name='q_input')
  w_input = Input(shape=(story_maxlen, len(vocab)), name='w_input')
  
  output = ControlUnit(d_dim=d)([c_input, q_input, w_input])
  controlUnitModel = Model(inputs = [c_input, q_input, w_input], output = output)

  controlUnitModel
'''
  
  
test_ControlUnit()