import time
import zipfile
import imageio
import json
from nlp import Vocabulary
from skimage import transform,io
import numpy as np

from keras.preprocessing.sequence import pad_sequences




def CLEVR_generator(dataset_split = 'val', batchSize = 3, maxLen = None):
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
        
        if len(batches['questions']) == batchSize:
            images = np.array(batches['images'])
            
            padded_questions = np.array( pad_sequences(batches['questions'], maxlen = maxLen) )
            padded_answers = np.array( pad_sequences(batches['answers'], maxlen = maxLen) )
            
            for key in ['images', 'questions', 'answers']:
                batches[key] = []

            yield [images, padded_questions], padded_answers
    
    

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
    
    batchSize = 11
    
    begin = time.time()
    generator = CLEVR_generator(batchSize = batchSize)
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
    test_generator()