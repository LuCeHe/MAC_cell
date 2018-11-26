# Copyright (c) 2018, 
#
# authors Luca Celotti
# while students at Université de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.



from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate, LSTM, Lambda, Reshape
from keras.layers import TimeDistributed, Bidirectional, Embedding, RepeatVector
import keras.backend as K

import tensorflow as tf

from MAC_variants import ControlUnit, ReadUnit, WriteUnit, MAC_layer, OutputUnit, \
                         completeMACmodel_simple
from nlp import generateBatchRandomQuestions


    
    
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
          

    print('')
    print('    Test Gradients')
    print('')
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        
    print('')
    print('    Test Fit')
    print('')
    
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    model.fit(input_data, c_i_1)


          
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
    


def test_ReadUnit_Gradients():
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

    #r_i = model.predict(input_data)
    #print(len(r_i))
    #print(r_i.shape)
    #print(r_i)     
    #print('')
    model.summary()    
    
    grad = tf.gradients(xs=[c_input, m_input, k_input], ys=model.output)
    print('grad:     ', grad)    
        
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    #for layer in model.layers:
    #    print(layer)
    
    print('')
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
    #weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
    gradients = model.optimizer.get_gradients(model.output, weights) # gradient tensors
    
    #print(weights)
    
    model.fit(input_data, c_i)

def test_WriteUnit():
    d = 3
    
    print('')
    print('    Test Predict')
    print('')
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
    
    print('')
    print('    Test Gradients')
    print('')
    weights = model.trainable_weights # weight tensors
    
    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        print(w)
        print('        ', g)  
        
    print('')
    print('    Test Fit')
    print('')
    
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    model.fit(input_data, m_i_1)
    
    




def get_inputs_MAC(d, 
                   batchSize, 
                   biLSTM = True, 
                   maxLen = None, 
                   embedding = False):
    
    # Inputs      
    question = generateBatchRandomQuestions(batchSize, maxLen)
    
    if not embedding:
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
        if not embedding:
            inputs_questions = Input(shape=(None,1), name='question')
        else:
            inputs_questions = Input(shape=(None,), name='question')
            
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
      

def test_pMAC(p=3):
    
    # parameters
    
    d = 2
    batchSize = 2

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(p):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)


    model = Model(inputs = input_layers, output = [c, m])
    
    #model.summary()
    c_i, m_i = model.predict(input_data)
    print(c_i)
    print('')
    print(m_i)
    

def test_pMAC_wOutput(p=3):
    
    # parameters
    
    d = 2
    batchSize = 2

    # get input data and input layers
    
    input_data, input_layers = get_inputs_MAC(d, batchSize)

    # Build model
    
    c, q_input, cws_input, m, k_input = input_layers
    
    for _ in range(p):
        c, m = MAC_layer(c, q_input, cws_input, m, k_input)



    softmax_output = OutputUnit(20)([m, q_input])
    model = Model(inputs = input_layers, output = [c, softmax_output])
    
    #model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i) 
    print('')
    print(softmax_output_i)


def test_pMAC_wO_wbiLSTM(p=3):
    
    # parameters
    
    d = 3
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

    def slice_questions(x, where):
        return x[:, where, :]
    
    # plug biLSTM    
    forward, backward = Bidirectional(LSTM(d, return_sequences=True), input_shape=(None, 1), merge_mode=None)(q_input)    
    
    # word representation
    cws = concatenate([forward, backward], axis=1)
    
    # sentence representation
    lenSentence = maxLen 
    fquestions = Lambda(slice_questions,arguments={'where':lenSentence-1})(cws)    #cws[:, lenSentence-1, :]
    bquestions = Lambda(slice_questions,arguments={'where':lenSentence})(cws)    #cws[:, lenSentence, :] 
    
    q = concatenate([fquestions, bquestions], axis=1)
    
    for _ in range(p):
        c, m = MAC_layer(c, q, cws, m, k_input)
    softmax_output = OutputUnit(20)([m, q])
    model = Model(inputs = input_layers, output = [c, softmax_output])    
    
    model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i) 
    print('')
    print(softmax_output_i)


def test_pMAC_wBiLSTM_wEmbedding(p=3):
    
    # parameters
    
    d = 3
    batchSize = 2
    maxLen = 10
    embDim = 5
    
    

    ########################################
    #    get input data and input layers
    ########################################
    
    input_data, input_layers = get_inputs_MAC(d, 
                                              batchSize, 
                                              biLSTM = False, 
                                              maxLen = maxLen,
                                              embedding = True)  
    c, q_input, m, k_input = input_layers
    
    ########################################
    #          Build model
    ########################################

    def slice_questions(x, where):
        return x[:, where, :]
    
    
    print(K.int_shape(q_input))
    embed = Embedding(102, embDim)(q_input)
    print(K.int_shape(embed))
            
    # plug biLSTM    
    forward, backward = Bidirectional(LSTM(d, return_sequences=True), 
                                      input_shape=(None, embDim), 
                                      merge_mode=None)(embed)    
    
    # word representation
    cws = concatenate([forward, backward], axis=1)
    
    # sentence representation
    lenSentence = maxLen 
    fquestions = Lambda(slice_questions,arguments={'where':lenSentence-1})(cws)    #cws[:, lenSentence-1, :]
    bquestions = Lambda(slice_questions,arguments={'where':lenSentence})(cws)    #cws[:, lenSentence, :] 
    
    q = concatenate([fquestions, bquestions], axis=1)
    
    for _ in range(p):
        c, m = MAC_layer(c, q, cws, m, k_input)
    softmax_output = OutputUnit(20)([m, q])
    model = Model(inputs = input_layers, output = [c, softmax_output])    
    
    model.summary()
    c_i, softmax_output_i = model.predict(input_data)
    print(c_i) 
    print('')
    print(softmax_output_i)

    
def test_ResNet50():
    
    d = 512
    
    from keras.applications.resnet50 import ResNet50
    from keras.layers import Input, Conv2D
        
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
    
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        print(layer.name)
        layer.trainable = False
    x = base_model.get_layer('activation_40').output
    
    x = Conv2D(d, (2, 2), padding="same")(x)
    
    x = Conv2D(d, (2, 2), padding="same")(x)
    
    model = Model(input_tensor, x)
    model.summary()
                


   
def test_simpleMAC(maxLen=None):
    MAC = completeMACmodel_simple(maxLen=maxLen)
    model = MAC.model()
    
    model.summary()
    
    ########################################
    #    get input data and input layers
    ########################################
    
    input_data, _ = get_inputs_MAC(d, 
                                   batchSize, 
                                   biLSTM = False, 
                                   maxLen = maxLen,
                                   embedding = True)  
    c, q_input, m, k_input = input_layers

    softmax = model.predict(input_data)


    
if __name__ == '__main__':
    
    test_ControlUnit()
    #test_ReadUnit()
    #test_WriteUnit()
    #test_MAC()
    #test_pMAC()
    #test_pMAC_wOutput()   
    #test_pMAC_wO_wbiLSTM()
    #test_ResNet50()
    ##test_pMAC_wBiLSTM_wEmbedding(
    #test_ReadUnit_Gradients()


    