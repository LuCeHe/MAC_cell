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
Created on Wed Nov  7 16:50:28 2018

@author: celottil
"""


import zipfile
import json
import numpy as np
from nltk.grammar import Nonterminal, CFG, Production


class Vocabulary(object):
    endToken = '<END>'
    
    def __init__(self, tokens):

        if Vocabulary.endToken in tokens:
            tokens.remove(Vocabulary.endToken)

        indicesByTokens = dict()
        tokens = [Vocabulary.endToken] + list(tokens)
        for i, token in enumerate(tokens):
            indicesByTokens[token] = i
        self.__dict__.update(tokens=tokens,
                             indicesByTokens=indicesByTokens)
        
    def __eq__(self, other):
        return self.tokens == other.tokens

    def __ne__(self, other):
        return self.tokens != other.tokens

    def __add__(self, other):
        # NOTE: ignore the end token
        tokens = set(self.tokens[1:])
        tokens.update(other.tokens[1:])
        return Vocabulary(list(sorted(tokens)))

    def sort(self):
        # NOTE: ignore the end token
        tokens = sorted(self.tokens[1:])
        self.indicesByTokens = dict()
        self.tokens = [Vocabulary.endToken] + tokens
        for i, token in enumerate(self.tokens):
            self.indicesByTokens[token] = i

    def indexToToken(self, idx):
        return self.tokens[idx]

    def indicesToTokens(self, indices, offset=0):
        #print(indices)
        #for i in indices:
        #    print(i)
        #    print(i, self.tokens[i-offset])
        return [self.tokens[i-offset] for i in indices]

    def tokenToIndex(self, token, offset=0):
        return self.indicesByTokens[token] + offset

    def tokensToIndices(self, tokens, offset=0):
        indices = []
        for token in tokens:
            indices.append(self.indicesByTokens[token] + offset)
        return indices

    def sentenceToIndices(self, sentence, offset=0):
        return self.tokensToIndices(sentenceSplit(sentence), offset)

    def sentencesToIndices(self, sentences, offset=0):
        indices = [self.sentenceToIndices(sentence, offset) for sentence in sentences]
        return indices

    def indicesToSentence(self, indices, offset=0):
        return ' '.join(self.indicesToTokens(indices, offset))
    
    def indicesToSentences(self, indices_list, offset=0):
        if type(indices_list).__module__ == 'numpy':
            indices_list = indices_list.tolist()
            
            # unpad:
            
            indices_list = [list(filter((0).__ne__, indices)) for indices in indices_list]
            
        sentences = [self.indicesToSentence(indices, offset) for indices in indices_list]
        return sentences

    @staticmethod
    def fromNltkGrammar(grammar):
        tokens = []
        for production in grammar.productions():
            for p in production.rhs():
                if not isinstance(p, Nonterminal):
                    tokens.append(p)

        # Remove redundant tokens and sort
        tokens = list(set(tokens))
        tokens.sort()

        return Vocabulary(tokens)
    
    @staticmethod
    def fromNpy(npy_filename = 'data/vocabularyComplete.npy'):
        tokens = np.load(npy_filename)
        return Vocabulary(tokens)
    
    def getMaxVocabularySize(self):
        return len(self.tokens)


def sentenceSplit(sentence):
    split_sentence = sentence.split()
    
    punctuation = ['.', ';', '?']
    
    for p in punctuation:
        is_p_contained = [p in token for token in split_sentence]
        for i, item in enumerate(is_p_contained):
            if item:
                
                word = split_sentence[i][:-1]
                punctuation_simbol = split_sentence[i][-1]
                
                split_sentence[i] = word
                split_sentence += [punctuation_simbol]
    
    return split_sentence
    
    
def build_CLEVR_Vocabulary():
    # TODO: get vocabulary from all datasets
    CLEVR_zip = zipfile.ZipFile("data/CLEVR_v1.0.zip", "r")
    
    sets_names = ['train', 'val', 'test']
    

    #logger.info('\n\nLoading .json ')
    
    vocabularyQuestions = []
    vocabularyAnswers = []
    for set_name in sets_names:
        
        print('\nLoading .json %s'%(set_name))
        
        json_filename = 'CLEVR_v1.0/questions/CLEVR_%s_questions.json'%(set_name)
        with CLEVR_zip.open(json_filename) as f:  
            data = f.read()  
            d = json.loads(data.decode("utf-8"))
    
        print('Building Vocabulary')
        
        i = 0 
        for example in d['questions']:
            i += 1
            
            question       = example['question']
            vocabularyQuestions += sentenceSplit(question)
            
            try:
                answer         = example['answer']
                vocabularyAnswers += sentenceSplit(answer)
            except KeyError:
                pass
            
            #if i>=3: break
        
    vocabularyQuestions = sorted(list(set(vocabularyQuestions)))
    vocabularyAnswers = sorted(list(set(vocabularyAnswers)))
    vocabularyComplete = sorted(list(set(vocabularyQuestions + vocabularyAnswers)))
    
    np.save('data/vocabularyQuestions.npy', vocabularyQuestions)
    np.save('data/vocabularyAnswers.npy', vocabularyAnswers)
    np.save('data/vocabularyComplete.npy', vocabularyComplete)
    
    print('Finished!')
    
    
if __name__ == '__main__':
    #build_CLEVR_Vocabulary()
    vocabulary = np.load('data/vocabularyComplete.npy')
    print(len(vocabulary))