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

"""
Created on Tue Nov  6 23:17:58 2018

@author: celottil


 - [DONE] read png
 - [DONE] read json
 - [DONE] read json line by line
 - [DONe] build generator
 - [DONE] get all the vocabulary
 - [DONE] tokenize generator
 - [DONE] batches
 - [DONE] tokenize questions differently than answers
 - [DONE] when tokenize, separate '?', ';', etc
 - [DONE] check why images have 4 dims
 - [DONE] paddled lanugage and images correct size
 - [DONE] plug BiLSTM with MAC
 - [DONE] plug RESNET50 with MAC
 - plug RESNET101 with MAC (converter)
 - allow for variable length sentences (maxLen = None)
 
 
"""

import logging
import numpy as np
import time
from nlp import Vocabulary

#import warnings
#warnings.filterwarnings("ignore")
    
from MAC_variants import completeMACmodel_simple

logger = logging.getLogger(__name__)




def train_MAC_on_CLEVR():
    
    vocabularyQuestions = Vocabulary.fromNpy('data/vocabularyQuestions.npy')
    vocabularyAnswers = Vocabulary.fromNpy('data/vocabularyAnswers.npy')
    inputVocabSize  = len(vocabularyQuestions.tokens)
    outputVocabSize = len(vocabularyAnswers.tokens)
        
    batchSize = 32
    maxLen = 10
    modelFilename = 'MAC'
    
    MAC = completeMACmodel_simple(d=512,  
                                  maxLen=maxLen, 
                                  p=12, 
                                  embDim=32,
                                  inputVocabSize=inputVocabSize,
                                  outputVocabSize=outputVocabSize)
    MAC.trainOnClevr(batchSize, modelFilename) 



    
if __name__ == '__main__':
    
    #test()
    #CLEVR_generator()
    #build_CLEVR_Vocabulary()
    train_MAC_on_CLEVR()