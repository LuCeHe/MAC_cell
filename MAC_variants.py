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
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#from keras import backend as K
import keras.backend.tensorflow_backend as K
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, concatenate, Input, Conv2D, Embedding, \
                            Bidirectional, concatenate, LSTM, Lambda, \
                            TimeDistributed, Bidirectional, RepeatVector
from keras.engine.topology import Layer
from keras import initializers

import tensorflow as tf

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
        
        assert input_shape[0][1] == input_shape[1][1]       # c_i_1 with q_i
        assert input_shape[0][1] == input_shape[2][2]       # c_i_1 with cws
        
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


def MAC_layer(d, c_i_1, q, cws, m_i_1, KB):
    
    print('c_i_1:   ', c_i_1)
    print('')
    print('q:       ', q)
    
    print('')
    print(K.shape(c_i_1)[0])
    # FIXME: plug again some of the following asssertions
    #assert 2*K.int_shape(c_i_1)[1] == K.int_shape(q)[1] 
    
    q_i = Dense(d, activation='linear')(q)
    c_i = ControlUnit()([c_i_1, q_i, cws])    
    r_i = ReadUnit()([c_i, m_i_1, KB])
    m_i = WriteUnit()([c_i, r_i, m_i_1])

    return [c_i, m_i]
  

class completeMACmodel_simple(object):
    
    ##############################################
    #    - k MAC
    #    - output and BiLSTM
    #    - ResNet50
    ##############################################
    
    def __init__(self,
                 d=2,  
                 maxLen=None, 
                 p=3, 
                 embDim=32):
        
        self.__dict__.update(d=d, 
                             maxLen=maxLen,               
                             p=p, 
                             embDim=embDim)

        self.build_model()
        
        
    def build_model(self):
        
        if self.maxLen == None:
            raise Exception("""Define maxLen for the questions, or figure
                            out how to fix it in the model to accept batches of
                            variable lengths""")
        
        
        ########################################
        #    get input layers
        ########################################
        
        input_images = Input(shape=(224, 224, 3), name='image')  
        input_questions = Input(shape=(None,), name='question')
        
        input_layers = [input_images, input_questions]
        
        def dynamic_zeros(x, d):
            b = tf.shape(x)[0]
            return tf.zeros(tf.stack([b, d]))
        
        c = Lambda(dynamic_zeros,arguments={'d': self.d})(input_images)
        m = Lambda(dynamic_zeros,arguments={'d': self.d})(input_images)
        
        ########################################
        #          Build model
        ########################################
    
    
        # ---------------- visual pipeline
        
        base_model = ResNet50(input_tensor=input_images, weights='imagenet', include_top=False)

        # Freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.get_layer('activation_40').output
        
        x = Conv2D(self.d, (3, 3), padding="same")(x)
        
        k_hwd = Conv2D(self.d, (3, 3), padding="same")(x)


        # ---------------- language pipeline
        
        embed = Embedding(102, self.embDim)(input_questions)
        
        # plug biLSTM    
        forward, backward = Bidirectional(LSTM(self.d, return_sequences=True), input_shape=(None, self.embDim), merge_mode=None)(embed)    
        
        # word representation
        cws = concatenate([forward, backward], axis=1)
        
        # sentence representation
        def slice_questions(x, where):
            return x[:, where, :]

        lenSentence = self.maxLen 
        fquestions = Lambda(slice_questions,arguments={'where':lenSentence-1})(cws)    #cws[:, lenSentence-1, :]
        bquestions = Lambda(slice_questions,arguments={'where':lenSentence})(cws)    #cws[:, lenSentence, :] 
        
        q = concatenate([fquestions, bquestions], axis=1)
        
        
        # ---------------- multimodal pipeline
        
        # k MAC cells
        for _ in range(self.p):
            c, m = MAC_layer(self.d, c, q, cws, m, k_hwd)
            
        softmax_output = OutputUnit(m, q)
        self.model = Model(inputs = input_layers, output = [softmax_output])    
        

    def passRandomNumpyThroughModel(self):
        pass
    
    def train(self):
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


if __name__ == '__main__':
    
    MAC = completeMACmodel_simple(maxLen=10)
    model = MAC.model
    
    model.summary()
    
    
