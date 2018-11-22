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


import time
import zipfile
import imageio
import json
from nlp import Vocabulary
from skimage import transform,io
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence


class CLEVR_sequence(Sequence):
    def __init__(self, dataset_split = 'val', batch_size = 3, maxLen = None):
        self.dataset_split, self.batch_size, self.maxLen = dataset_split, batch_size, maxLen

        self.CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
        
        if not dataset_split in ['train', 'test', 'val']:
            Exception("dataset_split parameter should be either 'train', 'test' or 'val'")
            
        json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(dataset_split)
        
        #logger.info('\n\nLoading .json ')
        #print('\n\nLoading .json ')
    
        
        with self.CLEVR_zip.open(json_filename) as f:  
            data = f.read()  
            self.d = json.loads(data.decode("utf-8"))
           
        
        self.vocabularyQuestions = Vocabulary.fromNpy('data/vocabularyQuestions.npy')
        self.vocabularyAnswers = Vocabulary.fromNpy('data/vocabularyAnswers.npy')
        self.outputVocabSize = len(self.vocabularyAnswers.tokens)
        
        
    def __len__(self):
        if self.dataset_split == 'train':
            length = 699989
        elif self.dataset_split == 'test':
            length = 149988
        elif self.dataset_split == 'val':
            length = 149991
        return int(np.ceil(length / float(self.batch_size)))

    def __getitem__(self, idx):
        #print('Generating Samples')    
        
        batches = {}
        for key in ['images', 'questions', 'answers']:
            batches[key] = []
            
        
        for example in self.d['questions']:
            
            image_filename = example['image_filename']
            image_filename = 'CLEVR_v1.0/images/' + self.dataset_split + '/' + image_filename
            imgfile = self.CLEVR_zip.read(image_filename)
            image = imageio.imread(imgfile)
            # read in grey-scale
            #grey = io.imread(imgfile)
            # resize to 28x28
            image = transform.resize(image, (228,228), mode='symmetric', preserve_range=True)
    
            
            batches['images'].append(image[:,:,:3])
            
            question       = example['question']
            question = self.vocabularyQuestions.sentenceToIndices(question)
            batches['questions'].append(question)
            
            answer         = example['answer']
            answer = self.vocabularyAnswers.sentenceToIndices(answer)
            batches['answers'].append(answer)
            
            if len(batches['questions']) >= self.batch_size:
                images = np.array(batches['images'])
                
                padded_questions = np.array( pad_sequences(batches['questions'], maxlen = self.maxLen) )
                categorical_answer = to_categorical(batches['answers'], num_classes=self.outputVocabSize)
    
                #padded_answers = np.array( pad_sequences(batches['answers'], maxlen = maxLen) )
                
                for key in ['images', 'questions', 'answers']:
                    batches[key] = []
    
                return [images, padded_questions], categorical_answer


def CLEVR_generator(dataset_split = 'val', batch_size = 3, maxLen = None):
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    if not dataset_split in ['train', 'test', 'val']:
        Exception("dataset_split parameter should be either 'train', 'test' or 'val'")
        
    json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(dataset_split)
    
    #logger.info('\n\nLoading .json ')
    #print('\n\nLoading .json ')

    
    with CLEVR_zip.open(json_filename) as f:  
        data = f.read()  
        d = json.loads(data.decode("utf-8"))
       
    
    vocabularyQuestions = Vocabulary.fromNpy('data/vocabularyQuestions.npy')
    vocabularyAnswers = Vocabulary.fromNpy('data/vocabularyAnswers.npy')
    outputVocabSize = len(vocabularyAnswers.tokens)
    
    #print('Generating Samples')    
    
    batches = {}
    for key in ['images', 'questions', 'answers']:
        batches[key] = []
        
    
    for example in d['questions']:
        
        image_filename = example['image_filename']
        image_filename = 'CLEVR_v1.0/images/' + dataset_split + '/' + image_filename
        imgfile = CLEVR_zip.read(image_filename)
        image = imageio.imread(imgfile)
        # read in grey-scale
        #grey = io.imread(imgfile)
        # resize to 28x28
        image = transform.resize(image, (228,228), mode='symmetric', preserve_range=True)

        
        batches['images'].append(image[:,:,:3])
        
        question       = example['question']
        question = vocabularyQuestions.sentenceToIndices(question)
        batches['questions'].append(question)
        
        answer         = example['answer']
        answer = vocabularyAnswers.sentenceToIndices(answer)
        batches['answers'].append(answer)
        
        if len(batches['questions']) >= batch_size:
            images = np.array(batches['images'])
            
            padded_questions = np.array( pad_sequences(batches['questions'], maxlen = maxLen) )
            categorical_answer = to_categorical(batches['answers'], num_classes=outputVocabSize)

            #padded_answers = np.array( pad_sequences(batches['answers'], maxlen = maxLen) )
            
            for key in ['images', 'questions', 'answers']:
                batches[key] = []

            yield [images, padded_questions], categorical_answer
    
def test_content_jsons():
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    for dataset_split in ['train', 'test', 'val']:
        
        json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(dataset_split)
                
        with CLEVR_zip.open(json_filename) as f:  
            data = f.read()  
            d = json.loads(data.decode("utf-8"))
        
        print('       ', dataset_split)
        print(d['info'])
        print(d['questions'][0].keys())
        print(len(d['questions']))
        print('')
       

def test():    
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    
    zip_subfolders = [x for x in CLEVR_zip.namelist() if x.endswith('/')]
    print(zip_subfolders)
    print('')
    
    questions = [x for x in CLEVR_zip.namelist() if 'questions' in x]
    
    print(questions)
    print('')
    imgfile = CLEVR_zip.read('CLEVR_v1.0/images/train/CLEVR_train_035898.png')
    
    
    im = imageio.imread(imgfile)
    print(type(im))
    print('')
    
        
    
    # FIXME: this following lines work, but too slow to load all the train data.
    # Find a way to load the json line by line, or dictionary key by key
    with CLEVR_zip.open('CLEVR_v1.0/questions/CLEVR_test_questions.json') as f:  
        data = f.read()  
        d = json.loads(data.decode("utf-8"))
        print(d['questions'][0])
    
    print('')
    print(d['info'])

def test_generator():
    
    batch_size = 11
    
    begin = time.time()
    generator = CLEVR_generator(batch_size = batch_size)
    end = time.time()
    LoadingTime = end - begin
    print('it took %ds to load'%(LoadingTime))
    
    
    nSamples = 3
    TotSamplingTime = 0
    for _ in range(nSamples):
        begin = time.time()
        batch = generator.__next__()
        end = time.time()
        SamplingTime = end - begin
        TotSamplingTime += SamplingTime
        for element in batch:
            if type(element) == list:
                for sub_element in element:
                    print(sub_element.shape)
            else:
                print(element.shape)
                
        print('in %ds'%(SamplingTime)) 
        print('')
        
    print('it took %ds/sample to sample'%(TotSamplingTime/nSamples))    



if __name__ == '__main__':
    #test_generator()
    test_content_jsons()