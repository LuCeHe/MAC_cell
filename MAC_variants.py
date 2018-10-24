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
  
  
    def __init__(self, **kwargs):
        
        super(ControlUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        input_dim = input_shape[0][1]
        assert input_shape[0][1] == input_shape[1][1]
        print(input_dim)
        
        self.d_dim = input_shape[0][1]
    
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
        super(ControlUnit, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i_1 = inputs[0]
        q = inputs[1]
        cw_s = inputs[2]
    
    
        print('c_i_1:   ', c_i_1)
        print('q:       ', q)
        print('cw_s:    ', cw_s)
        
        
        # equation c1  
        conc_cq = K.concatenate([c_i_1, q], axis=1)
        print('conc_cq: ', K.int_shape(conc_cq))
    
        cq_i = K.dot(conc_cq, K.transpose(self.W_d2d))
        cq_i = K.bias_add(cq_i, self.b_d, data_format=None)  
        
        print('cq_i:    ', K.int_shape(cq_i))
    
        # equation c2.1  
        cqcw = cw_s * cq_i
        print('cw_s:    ', K.int_shape(cw_s))
        print('cqcw:    ', K.int_shape(cqcw))
    
        ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
        
        print('ca_is:   ', K.int_shape(ca_is))
    
        # equation c2.2
        cv_is = K.softmax(ca_is)
        print('cv_is:   ', K.int_shape(cv_is))
    
        # equation c2.3
        c_i = K.sum(cv_is*cw_s, axis=1)
        
        print('c_i:     ', K.int_shape(c_i))
    
        return c_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  

class writeUnit(Layer):
    pass



class ReadUnit(Layer):

      
    def __init__(self, **kwargs):
        
        super(ReadUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        input_dim = input_shape[0][1]
        assert input_shape[0][1] == input_shape[1][1]
        print(input_dim)
        
        self.d_dim = input_shape[0][1]
    
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
        super(ControlUnit, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i_1 = inputs[0]
        q = inputs[1]
        cw_s = inputs[2]
    
    
        print('c_i_1:   ', c_i_1)
        print('q:       ', q)
        print('cw_s:    ', cw_s)
        
        
        # equation c1  
        conc_cq = K.concatenate([c_i_1, q], axis=1)
        print('conc_cq: ', K.int_shape(conc_cq))
    
        cq_i = K.dot(conc_cq, K.transpose(self.W_d2d))
        cq_i = K.bias_add(cq_i, self.b_d, data_format=None)  
        
        print('cq_i:    ', K.int_shape(cq_i))
    
        # equation c2.1  
        cqcw = cw_s * cq_i
        print('cw_s:    ', K.int_shape(cw_s))
        print('cqcw:    ', K.int_shape(cqcw))
    
        ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
        
        print('ca_is:   ', K.int_shape(ca_is))
    
        # equation c2.2
        cv_is = K.softmax(ca_is)
        print('cv_is:   ', K.int_shape(cv_is))
    
        # equation c2.3
        c_i = K.sum(cv_is*cw_s, axis=1)
        
        print('c_i:     ', K.int_shape(c_i))
    
        return c_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  
    
    
    
    
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
    
    
    question_1 = np.array([1, 2, 5, 2, 4, 9, 1])
    n_timesteps = len(question_1)
    question_1 = question_1.reshape(1, n_timesteps, 1)
    question_2 = np.array([1, 2, 5, 2])
    n_timesteps = len(question_2)
    question_2 = question_2.reshape(1, n_timesteps, 1)
  
    
    for question in [question_1]:
      
        # biLSTM
        # NOTE: if return_sequences=False and 'concat', it gives directly q 
        # in the exact way we need it, but it doesn't give cw, that you get it 
        # with if return_sequences=True and None
        
        inputs = Input(shape=(None,1), name='question')
        output = Bidirectional(LSTM(d, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=None)(inputs)
        
        biLSTM_Model = Model(inputs = inputs, output = output)
        cw = biLSTM_Model.predict(question)


        #for cwi in cw:
        #    print(cwi)
        #    print('')            
        #print('')
        #print('')
        cws = np.concatenate(cw, axis=1)
        print(cws)
        #print('')

        q = [cw[1][0][0].tolist() + cw[0][0][-1].tolist()]
        q = np.array(q)
        print(q)
        print('')
        
        # position aware vector      
        inputs = Input(shape=(2*d,), name='question')
        output = Dense(d, activation='linear')(inputs)
      
        q_i_Model = Model(inputs = inputs, output = output)
        q_i = q_i_Model.predict(q)
        
        # c_i_1                
        c_i_1 = np.random.uniform(0, 1, size=(1,d))
        
        # test ControlUnit without training        
        input_data = [c_i_1, q_i, cws]
  
  
        # build model  
  
  
        c_input = Input(shape=(d,), name='c_input')
        q_input = Input(shape=(d,), name='q_input')
        w_input = Input(shape=(None,d), name='w_input')
  
        output = ControlUnit()([c_input, q_input, w_input])
        controlUnitModel = Model(inputs = [c_input, q_input, w_input], output = output)

        c_i = controlUnitModel.predict(input_data)
        print(c_i)
          
          
  
test_ControlUnit()