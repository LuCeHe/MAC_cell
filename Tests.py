#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:05:01 2018

@author: hey
"""


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np


from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from MAC_variants import ControlUnit, ReadUnit, WriteUnit, MAC_layer, OutputUnit

def test_ControlUnit():
    d = 2
    
    
    question_1 = np.array([1, 2, 5, 2, 4, 9, 1])
    n_timesteps = len(question_1)
    question_1 = question_1.reshape(1, n_timesteps, 1)
    question_2 = np.array([1, 2, 5, 2])
    n_timesteps = len(question_2)
    question_2 = question_2.reshape(1, n_timesteps, 1)
  
    # FIXME: I don't know if it works with a batchSize = 2    
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
        model = Model(inputs = [c_input, q_input, w_input], output = output)

        c_i = model.predict(input_data)
        print(c_i)
          
          
def test_ReadUnit():
    d = 3
    
    
    # data
    
    # FIXME: it doesn't work with a batchSize = 2
    batchSize = 2
    c_i = np.random.uniform(0, 1, size=(batchSize,d))
    m_i_1 = np.random.uniform(0, 1, size=(batchSize,d))
    k_hw = np.random.uniform(0, 1, size=(batchSize, 3, 4, d))
    
    input_data = [c_i, m_i_1, k_hw]
  
  
    # build model  
    c_input = Input(shape=(d,), name='c_input')
    m_input = Input(shape=(d,), name='q_input')
    k_input = Input(shape=(None, None, d), name='w_input')
  
    output = ReadUnit()([c_input, m_input, k_input])
    model = Model(inputs =[c_input, m_input, k_input], output = output)

    r_i = model.predict(input_data)
    print(len(r_i))
    print(r_i.shape)
    print(r_i)     
    

def test_WriteUnit():
    d = 3
    
    
    # data
    
    # FIXME: it doesn't work with a batchSize = 2
    batchSize = 1
    c_i = np.random.uniform(0, 1, size=(batchSize, d))
    r_i = np.random.uniform(0, 1, size=(batchSize, d))
    m_i_1 = np.random.uniform(0, 1, size=(batchSize, d))
    
    input_data = [c_i, r_i, m_i_1]
  
  
    # build model  
    c_input = Input(shape=(d,), name='c_input')
    r_input = Input(shape=(d,), name='r_input')
    m_input = Input(shape=(d,), name='m_input')
  
    output = WriteUnit()([c_input, r_input, m_input])
    model = Model(inputs =[c_input, r_input, m_input], output = output)

    m_i = model.predict(input_data)
    print(len(m_i))
    print(m_i.shape)
    print(m_i)




def get_inputs_MAC(d, batchSize):
    
    # Inputs
    
    question = np.array([1, 2, 5, 2, 4, 9, 1])  #np.array([1, 2, 5, 2])
    n_timesteps = len(question)
    question = question.reshape(1, n_timesteps, 1)
          
    # biLSTM
    inputs = Input(shape=(None,1), name='question')
    output = Bidirectional(LSTM(d, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=None)(inputs)
    
    biLSTM_Model = Model(inputs = inputs, output = output)
    cw = biLSTM_Model.predict(question)
    cws = np.concatenate(cw, axis=1)

    q = [cw[1][0][0].tolist() + cw[0][0][-1].tolist()]
    q = np.array(q)
    
    # c_i_1                
    c_i_1 = np.random.uniform(0, 1, size=(1,d))
    k_hw = np.random.uniform(0, 1, size=(batchSize, 3, 4, d))
    m_i_1 = np.random.uniform(0, 1, size=(batchSize,d))
    
    # test ControlUnit without training        
    input_data = [c_i_1, q, cws, m_i_1, k_hw]
  
    
    
    # Build model input 
    
    c_input = Input(shape=(d,), name='c_input')
    q_input = Input(shape=(2*d,), name='q_input')
    cws_input = Input(shape=(None,d), name='cws_input')
    m_input = Input(shape=(d,), name='m_input')
    k_input = Input(shape=(None, None, d), name='k_input')

    input_layers = [c_input, q_input, cws_input, m_input, k_input]
    return input_data, input_layers


def test_MAC():
    
    # parameters
    
    d = 2
    batchSize = 1

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, m = MAC_layer(*input_layers)
    model = Model(inputs = input_layers, output = [c, m])

    c_i, m_i = model.predict(input_data)
    print(c_i, m_i)
      

def test_kMAC(k=3):
    
    # parameters
    
    d = 2
    batchSize = 1

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(k):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)


    model = Model(inputs = input_layers, output = [c, m])
    
    model.summary()
    c_i, m_i = model.predict(input_data)
    print(c_i, m_i)
    

def test_kMAC_wOutput(k=3):
    
    # parameters
    
    d = 2
    batchSize = 1

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(k):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)



    softmax_output = OutputUnit(m, q_input)
    model = Model(inputs = input_layers, output = [c, softmax_output])
    
    model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i, softmax_output_i)


    
if __name__ == '__main__':
    
    test_kMAC_wOutput()