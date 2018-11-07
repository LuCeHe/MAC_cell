#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:17:58 2018

@author: celottil


 - [DONE] read png
 - [DONE] read json
 - [DONE] read json line by line
 - build generator
 - get all the vocabulary
 - tokenize generator
 - batches
"""

import zipfile
import imageio
import json
import logging

logger = logging.getLogger(__name__)


def build_CLEVR_Vocabulary():
    pass


def CLEVR_generator(dataset_split = 'train', batchSize = 2):
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    if not dataset_split in ['train', 'test', 'val']:
        Exception("dataset_split parameter should be either 'train', 'test' or 'val'")
        
    json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(dataset_split)
    
    #logger.info('\n\nLoading .json ')
    print('\n\nLoading .json ')

    
    with CLEVR_zip.open(json_filename) as f:  
        data = f.read()  
        d = json.loads(data.decode("utf-8"))
       
    print('Loaded')
    print('Building Vocabulary')
    vocabularyQuestions = []
    vocabularyAnswers = []
    
    i = 0 
    for example in d['questions']:
        i += 1
        
        question       = example['question']
        answer         = example['answer']
        
        vocabularyQuestions += question.split()
        vocabularyAnswers += answer.split()
        
        if i>=3: break
        
    print(vocabularyQuestions)
    vocabularyQuestions = sorted(list(set(vocabularyQuestions)))
    print(vocabularyQuestions)
    print(vocabularyAnswers)
    print('')
    print(d.keys)
    print('')
    print(d['info'])
    print('')
    vocabularyComplete = []
    
    print('Generating Samples')    
    for example in d['questions']:
        
        image_filename = example['image_filename']
        image_filename = 'CLEVR_v1.0/images/' + dataset_split + '/' + image_filename
        
        question       = example['question']
        answer         = example['answer']
        
        
        print('')
        print(image_filename)
        print(question)
        print(answer)
        print('')
        break
    
    

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
    CLEVR_generator()