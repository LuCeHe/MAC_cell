#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:17:58 2018

@author: celottil


 - [DONE] read png
 - [DONE] read json
 - [DONE] read json line by line
 - [DONe] build generator
 - [DONE] get all the vocabulary
 - tokenize generator
 - batches
 - tokenize questions differently than answers
 - [DONE] when tokenize, separate '?', ';', etc
"""

import zipfile
import imageio
import json
import logging
import numpy as np
import time
from nlp import Vocabulary


logger = logging.getLogger(__name__)






def CLEVR_generator(dataset_split = 'train', batchSize = 3):
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    if not dataset_split in ['train', 'test', 'val']:
        Exception("dataset_split parameter should be either 'train', 'test' or 'val'")
        
    json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(dataset_split)
    
    #logger.info('\n\nLoading .json ')
    print('\n\nLoading .json ')

    
    with CLEVR_zip.open(json_filename) as f:  
        data = f.read()  
        d = json.loads(data.decode("utf-8"))
       
    
    vocabularyQuestions = Vocabulary.fromNpy('data/vocabularyQuestions.npy')
    vocabularyAnswers = Vocabulary.fromNpy('data/vocabularyAnswers.npy')
    
    print('Generating Samples')    
    
    batches = {}
    for key in ['images', 'questions', 'answers']:
        batches[key] = []
        
        
    for example in d['questions']:
        
        image_filename = example['image_filename']
        image_filename = 'CLEVR_v1.0/images/' + dataset_split + '/' + image_filename
        imgfile = CLEVR_zip.read(image_filename)
        image = imageio.imread(imgfile)
        batches['images'].append(image)
        
        question       = example['question']
        question = vocabularyQuestions.sentenceToIndices(question)
        question = np.array(question)
        batches['questions'].append(question)
        
        answer         = example['answer']
        answer = vocabularyAnswers.sentenceToIndices(answer)
        answer = np.array(answer)
        batches['answers'].append(answer)
        
        if len(batches['questions']) == batchSize:
            images = np.array(batches['images'])
            questions = np.array(batches['questions'])
            answers = np.array(batches['answers'])
            
            for key in ['images', 'questions', 'answers']:
                batches[key] = []
            yield images, questions, answers
    
    

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

if __name__ == '__main__':
    
    
    #test()
    #CLEVR_generator()
    #build_CLEVR_Vocabulary()
    begin = time.time()
    generator = CLEVR_generator()
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
        print([element.shape for element in batch])
        print('in %ds'%(SamplingTime)) 
        print('')
        
    print('it took %ds/sample to sample'%(TotSamplingTime/nSamples))    
    