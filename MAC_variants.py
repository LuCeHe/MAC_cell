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

import numpy as np


'''
- [DONE] Control Unit
- [DONE] Control Unit test
- [DONE] Read Unit
- [DONE] Read Unit test
- Write Unit
- Write Unit test
- Mac unit
- check that they work on batches
- CLEVR
- CLEAR
- recurrent MAC
- goedel machine
'''

# MAC cell
# https://arxiv.org/pdf/1803.03067.pdf
class ControlUnit(Layer):
  
  
    def __init__(self, **kwargs):
        
        super(ControlUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        assert input_shape[0][1] == input_shape[1][1]
        
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
        
        # equation c1  
        conc_cq = K.concatenate([c_i_1, q], axis=1)
    
        cq_i = K.dot(conc_cq, K.transpose(self.W_d2d))
        cq_i = K.bias_add(cq_i, self.b_d, data_format=None)  
    
        # equation c2.1  
        cqcw = cw_s * cq_i

        ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
            
        # equation c2.2
        cv_is = K.softmax(ca_is)
    
        # equation c2.3
        c_i = K.sum(cv_is*cw_s, axis=1)
    
        return c_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  

class writeUnit(Layer):
    pass



class ReadUnit(Layer):

      
    def __init__(self, **kwargs):
        
        super(ReadUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        
        assert len(input_shape[2]) == 4
        assert input_shape[0][1] == input_shape[1][1]
        assert input_shape[0][1] == input_shape[2][3]

        
        self.d_dim = input_shape[0][1]
        
        # initial values of the learning parameters
        initial_W_ddm_value = np.random.uniform(0, 1, size=[self.d_dim, self.d_dim]) 
        initial_b_dm_value = np.random.uniform(0, 1, size=[self.d_dim])
        
        initial_W_ddk_value = np.random.uniform(0, 1, size=[self.d_dim, self.d_dim]) 
        initial_b_dk_value = np.random.uniform(0, 1, size=[self.d_dim])
        
        initial_W_d2d_value = np.random.uniform(0, 1, size=[self.d_dim, 2*self.d_dim])
        initial_b_d1_value = np.random.uniform(0, 1, size=[self.d_dim])
    
        initial_W_dd_value = np.random.uniform(0, 1, size=[self.d_dim, self.d_dim])
        initial_b_d2_value = np.random.uniform(0, 1, size=[self.d_dim])



        self.W_ddm = K.variable(initial_W_ddm_value)
        self.b_dm  = K.variable(initial_b_dm_value)

        self.W_ddk = K.variable(initial_W_ddk_value)
        self.b_dk  = K.variable(initial_b_dk_value)

        self.W_d2d = K.variable(initial_W_d2d_value)
        self.b_d1   = K.variable(initial_b_d1_value)
    
        self.W_dd  = K.variable(initial_W_dd_value)
        self.b_d2   = K.variable(initial_b_d2_value)

        self.trainable_weights = [self.W_ddm, self.b_dm, self.W_ddk, self.b_dk,
                                  self.W_d2d, self.b_d1, self.W_dd, self.b_d2]
        super(ReadUnit, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i = inputs[0]
        m_i_1 = inputs[1]
        k_hw = inputs[2]
    
    
        print('c_i:     ', c_i)
        print('m_i_1:   ', m_i_1)
        print('k_hw:    ', k_hw)
        
        
        # equation r1
        
        Wm_b = K.dot(m_i_1, K.transpose(self.W_ddm))
        Wm_b = K.bias_add(Wm_b, self.b_dm, data_format=None)  
        Wk_b = K.dot(m_i_1, K.transpose(self.W_ddk))
        Wk_b = K.bias_add(k_hw, self.b_dk, data_format=None)  
        
        print('Wm_b:    ', K.int_shape(Wm_b))
        print('Wk_b:    ', K.int_shape(Wk_b))
        
        # TODO: check the following multiplication is done properly
        I_ihw = Wm_b*Wk_b
        
        print('I_ihw:   ', K.int_shape(I_ihw))
        
        # equation r2
        conc_Ik = K.concatenate([I_ihw, k_hw], axis=3)
        
        print('conc_Ik: ', K.int_shape(conc_Ik))
        
        II_ihw = K.dot(conc_Ik, K.transpose(self.W_d2d))
        II_ihw = K.bias_add(II_ihw, self.b_d1, data_format=None) 
        
        print('II_ihw:  ', K.int_shape(II_ihw))
        
        
        # equation r31
        cI = c_i*II_ihw
        
        
        print('cI:      ', K.int_shape(cI))
        
        ra_ihw = K.dot(cI, K.transpose(self.W_dd))
        ra_ihw = K.bias_add(ra_ihw, self.b_d2, data_format=None) 
        
        
        print('ra_ihw:  ', K.int_shape(ra_ihw))
        
        # equation r32
        
        rv_ihw = K.softmax(ra_ihw)
        
        print('rv_ihw:  ', K.int_shape(rv_ihw))
        
        # equation r33 
        
        r_i = K.sum(rv_ihw*k_hw, axis=1)
        
        print('r_i:      ', K.int_shape(r_i))
        
        r_i = K.sum(r_i, axis=1)
        
        print('r_i:      ', K.int_shape(r_i))

        return r_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  
    
    
class WriteUnit(Layer):
  
  
    def __init__(self, **kwargs):
        
        super(WriteUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        assert input_shape[0][1] == input_shape[1][1]
        assert input_shape[0][1] == input_shape[2][1]
        
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
        super(WriteUnit, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i_1 = inputs[0]
        q = inputs[1]
        cw_s = inputs[2]
        
        # equation c1  
        conc_cq = K.concatenate([c_i_1, q], axis=1)
    
        cq_i = K.dot(conc_cq, K.transpose(self.W_d2d))
        cq_i = K.bias_add(cq_i, self.b_d, data_format=None)  
    
        # equation c2.1  
        cqcw = cw_s * cq_i

        ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
            
        # equation c2.2
        cv_is = K.softmax(ca_is)
    
        # equation c2.3
        c_i = K.sum(cv_is*cw_s, axis=1)
    
        return c_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  

class writeUnit(Layer):
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


