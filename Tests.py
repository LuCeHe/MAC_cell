#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:05:01 2018

@author: hey
"""


from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np


from keras.models import Sequential, Model
from keras.layers import LSTM
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

from MAC_variants import ControlUnit, ReadUnit, WriteUnit, MAC_layer, OutputUnit


def generateBatchRandomQuestions(batchSize, maxLen):
    
    questions = []
    for _ in range(batchSize):
        sentence_length = np.random.choice(9)
        randomQ = np.random.choice(7, sentence_length)
        questions.append(randomQ)
    
        
        
    padded_questions = pad_sequences(questions, maxlen = maxLen)
    
    print('Padded question')
    print('')
    print(padded_questions)
    print('')
    
    return padded_questions
    
    
def test_ControlUnit():
    d = 3
    batchSize = 4
    maxLen = None
        
    question = generateBatchRandomQuestions(batchSize, maxLen)
    question = question.reshape(batchSize, question.shape[1], 1)
    
    inputs = Input(shape=(None,1), name='question')
    output = Bidirectional(LSTM(d, return_sequences=True), input_shape=(None, 1), merge_mode=None)(inputs)
    
    biLSTM_Model = Model(inputs = inputs, output = output)
    forward, backward = biLSTM_Model.predict(question)
    
    # word representation
    cws = np.concatenate([forward, backward], axis=1)
    
    # sentence representation
    lenSentence = int(cws.shape[1]/2) 
    fquestions = cws[:, lenSentence-1, :]
    bquestions = cws[:, lenSentence, :]  
    q = np.concatenate([fquestions, bquestions], axis=1)
    
    # position aware vector      
    inputs = Input(shape=(2*d,), name='question')
    output = Dense(d, activation='linear')(inputs)
  
    q_i_Model = Model(inputs = inputs, output = output)
    q_i = q_i_Model.predict(q)
    
    # c_i_1                
    c_i_1 = np.random.uniform(0, 1, size=(batchSize,d))
    
    # test ControlUnit without training        
    input_data = [c_i_1, q_i, cws]
  
  
    # build model  
  
  
    c_input = Input(shape=(d,), name='c_input')
    q_input = Input(shape=(d,), name='q_input')
    w_input = Input(shape=(None,d), name='w_input')
  
    output = ControlUnit()([c_input, q_input, w_input])
    model = Model(inputs = [c_input, q_input, w_input], output = output)

    c_i = model.predict(input_data)
    print('c_i:     ', c_i)
          
          
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
    batchSize = 3
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




def get_inputs_MAC(d, batchSize, biLSTM = True, maxLen = None):
    
    # Inputs      
    question = generateBatchRandomQuestions(batchSize, maxLen)
    question = question.reshape(batchSize, question.shape[1], 1)
    
    
    if biLSTM:
        inputs = Input(shape=(None,1), name='question')
        output = Bidirectional(LSTM(d, return_sequences=True), input_shape=(None, 1), merge_mode=None)(inputs)
        
        biLSTM_Model = Model(inputs = inputs, output = output)
        forward, backward = biLSTM_Model.predict(question)
        
        # word representation
        cws = np.concatenate([forward, backward], axis=1)
        
        # sentence representation
        lenSentence = int(cws.shape[1]/2) 
        fquestions = cws[:, lenSentence-1, :]
        bquestions = cws[:, lenSentence, :]  
        q = np.concatenate([fquestions, bquestions], axis=1)
    
    # c_i_1                
    c_i_1 = np.random.uniform(0, 1, size=(batchSize,d))
    k_hw = np.random.uniform(0, 1, size=(batchSize, 3, 4, d))
    m_i_1 = np.random.uniform(0, 1, size=(batchSize,d))
    
    # Build model input 
    
    c_input = Input(shape=(d,), name='c_input')
    q_input = Input(shape=(2*d,), name='q_input')
    cws_input = Input(shape=(None,d), name='cws_input')
    m_input = Input(shape=(d,), name='m_input')
    k_input = Input(shape=(None, None, d), name='k_input')


    if biLSTM:
        input_layers = [c_input, q_input, cws_input, m_input, k_input]
        input_data = [c_i_1, q, cws, m_i_1, k_hw]

    else:
        inputs_questions = Input(shape=(None,1), name='question')
        input_layers = [c_input, inputs_questions, m_input, k_input]
        input_data = [c_i_1, question, m_i_1, k_hw]


    return input_data, input_layers


def test_MAC():
    
    # parameters
    
    d = 3
    batchSize = 4

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)
    
    shapes = [data.shape for data in input_data]
    print(shapes)
    # Build model
    
    c, m = MAC_layer(*input_layers)
    model = Model(inputs = input_layers, output = [c, m])

    c_i, m_i = model.predict(input_data)
    print(c_i)
    print('')
    print(m_i)
      

def test_kMAC(k=3):
    
    # parameters
    
    d = 2
    batchSize = 2

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(k):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)


    model = Model(inputs = input_layers, output = [c, m])
    
    #model.summary()
    c_i, m_i = model.predict(input_data)
    print(c_i)
    print('')
    print(m_i)
    

def test_kMAC_wOutput(k=3):
    
    # parameters
    
    d = 2
    batchSize = 2

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(k):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)



    softmax_output = OutputUnit(m, q_input)
    model = Model(inputs = input_layers, output = [c, softmax_output])
    
    #model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i) 
    print('')
    print(softmax_output_i)


def test_kMAC_wO_wbiLSTM(k=3):
    
    # parameters
    
    d = 2
    batchSize = 2
    maxLen = 10


    ########################################
    #    get input data and input layers
    ########################################
    
    input_data, input_layers = get_inputs_MAC(d, batchSize, biLSTM = False, maxLen = maxLen)  
    c, q_input, m, k_input = input_layers
    
    ########################################
    #          Build model
    ########################################

    # plug biLSTM
    
    forward, backward = Bidirectional(LSTM(d, return_sequences=True), input_shape=(None, 1), merge_mode=None)(q_input)
    
    print(K.int_shape(forward))
    
    # word representation
    cws = K.concatenate([forward, backward], axis=1)
    
    # sentence representation
    lenSentence = maxLen 
    fquestions = cws[:, lenSentence-1, :]
    bquestions = cws[:, lenSentence, :]  
    q = K.concatenate([fquestions, bquestions], axis=1)
    
    
    for _ in range(k):
        c, m = MAC_layer(c, q, cws, m, k_input)

    softmax_output = OutputUnit(m, q_input)
    model = Model(inputs = input_layers, output = [c, softmax_output])
    
    #model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i) 
    print('')
    print(softmax_output_i)

    
if __name__ == '__main__':
    
    #test_ControlUnit()
    #test_ReadUnit()
    #test_WriteUnit()
    #test_MAC()
    #test_kMAC()
    #test_kMAC_wOutput()   
    test_kMAC_wO_wbiLSTM()
    
    