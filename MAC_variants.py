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
from keras.layers import Dense, concatenate
from keras.engine.topology import Layer

import numpy as np


'''
- [DONE] Control Unit
- [DONE] Control Unit test
- [DONE] Read Unit
- [DONE] Read Unit test
- [DONE] Write Unit
- [DONE] Write Unit test
- [DONE] Mac unit
- [DONE] check that they work on batches
- [DONE] check if the repeated matrices in each equation
share parameters or not
- CLEVR
- implement self-attention:
    - as they do
    - as Graph Attention Net
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
        cq_i = K.expand_dims(cq_i, axis=1)
    
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
        
        
        # equation r1        
        Wm_b = K.dot(m_i_1, K.transpose(self.W_ddm))
        Wm_b = K.bias_add(Wm_b, self.b_dm, data_format=None)  
        Wk_b = K.dot(m_i_1, K.transpose(self.W_ddk))
        Wk_b = K.bias_add(k_hw, self.b_dk, data_format=None)          
        
        Wm_b = K.expand_dims(Wm_b,axis=1)
        Wm_b = K.expand_dims(Wm_b,axis=1)
        I_ihw = Wm_b*Wk_b        
        
        # equation r2
        conc_Ik = K.concatenate([I_ihw, k_hw], axis=3)        
        
        II_ihw = K.dot(conc_Ik, K.transpose(self.W_d2d))
        II_ihw = K.bias_add(II_ihw, self.b_d1, data_format=None) 
        
        # equation r31
        c_i = K.expand_dims(c_i,axis=1)
        c_i = K.expand_dims(c_i,axis=1)
        cI = c_i*II_ihw
        
        ra_ihw = K.dot(cI, K.transpose(self.W_dd))
        ra_ihw = K.bias_add(ra_ihw, self.b_d2, data_format=None)                 
        
        # equation r32        
        rv_ihw = K.softmax(ra_ihw)        
        
        # equation r33         
        r_i = K.sum(rv_ihw*k_hw, axis=1)        
        r_i = K.sum(r_i, axis=1)        
        return r_i
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim
  
    
    
class WriteUnit(Layer):
  
  
    def __init__(self, integration_steps = 1, **kwargs):
        self.integration_steps = integration_steps
        super(WriteUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        #print(input_shape)
        assert input_shape[0][1] == input_shape[1][1]
        assert input_shape[0][1] == input_shape[2][1]
        
        self.d_dim = input_shape[0][1]
    
        initial_W_d2d_value = np.random.uniform(0, 1, size=[self.d_dim, 2*self.d_dim])
        initial_b_d_value = np.random.uniform(0, 1, size=[self.d_dim])
    
        initial_W_1d1_value = np.random.uniform(0, 1, size=[1, self.d_dim])
        initial_b_11_value = np.random.uniform(0, 1, size=[1])
        
        if not self.integration_steps == 1:
            print('The optional features of the WU were not implemented yet!')
        #initial_W_dds_value = np.random.uniform(0, 1, size=[self.d_dim, self.d_dim])
        #initial_W_ddp_value = np.random.uniform(0, 1, size=[self.d_dim, self.d_dim])
        #initial_b_dsp_value = np.random.uniform(0, 1, size=[self.d_dim])

        #initial_W_1d2_value = np.random.uniform(0, 1, size=[1, self.d_dim])
        #initial_b_12_value = np.random.uniform(0, 1, size=[1])

        self.W_d2d = K.variable(initial_W_d2d_value)
        self.b_d   = K.variable(initial_b_d_value)
    
        self.W_1d1 = K.variable(initial_W_1d1_value)
        self.b_11  = K.variable(initial_b_11_value)

        #self.W_dds  = K.variable(initial_W_dds_value)
        #self.W_ddp  = K.variable(initial_W_ddp_value)
        #self.b_dsp   = K.variable(initial_b_dsp_value)

        #self.W_1d2 = K.variable(initial_W_1d2_value)
        #self.b_12  = K.variable(initial_b_12_value)

        self.trainable_weights = [self.W_d2d, self.b_d, self.W_1d1, self.b_11]
                                  #self.W_dds, self.W_ddp, self.b_dsp, self.W_1d2, self.b_12]
        
        super(WriteUnit, self).build(input_shape)
    
  
  
    def call(self, inputs, mask=None):
        c_i = inputs[0]
        r_i = inputs[1]
        m_i_1 = inputs[2]
        
        # equation c1  
        m_info = K.concatenate([r_i, m_i_1], axis=1)
        m_info = K.dot(m_info, K.transpose(self.W_d2d))
        m_info = K.bias_add(m_info, self.b_d, data_format=None)  
        
        #print('m_info:      ', K.int_shape(m_info))
    
        # equation c2.1  
        #cqcw = cw_s * cq_i

        #ca_is = K.dot(cqcw, K.transpose(self.W_1d))
        #ca_is = K.bias_add(ca_is, self.b_1, data_format=None)
            
        # equation c2.2
        #cv_is = K.softmax(ca_is)
    
        # equation c2.3
        #c_i = K.sum(cv_is*cw_s, axis=1)
    
        return m_info
  
  
    def get_output_shape_for(self, input_shape):
        return self.d_dim


def OutputUnit(m_p, q, num_softmax = 20):
    d = K.int_shape(m_p)[1]
    
    assert 2*K.int_shape(m_p)[1] == K.int_shape(q)[1]
    x = concatenate([m_p, q])
    x = Dense(d, activation='relu')(x)
    softmax_ouput_layer = Dense(num_softmax, activation='softmax')(x)
    
    return softmax_ouput_layer


def MAC_layer(c_i_1, q, cws, m_i_1, KB):
    
    assert 2*K.int_shape(c_i_1)[1] == K.int_shape(q)[1] 
    assert K.int_shape(c_i_1)[1] == K.int_shape(cws)[2]
    assert K.int_shape(c_i_1)[1] == K.int_shape(m_i_1)[1]
    assert K.int_shape(c_i_1)[1] == K.int_shape(KB)[3]
    
    d = K.int_shape(c_i_1)[1]
    q_i = Dense(d, activation='linear')(q)
    c_i = ControlUnit()([c_i_1, q_i, cws])    
    r_i = ReadUnit()([c_i, m_i_1, KB])
    m_i = WriteUnit()([c_i, r_i, m_i_1])
    return [c_i, m_i]
    

    
    
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


