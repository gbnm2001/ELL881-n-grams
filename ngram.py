import numpy as np
from threading import Thread
import pickle
from joblib import Parallel, delayed
import time
import math
class ThreadWithReturn(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class ngram:
    def __init__(self, n):
        self.N = n
        #two dimensional dictionary where 1st key is the condition and second key is the value
        #in case of unigram condition is null i.e., there is only one first key
        self.vocabulary = {}
        self.contexts = {}
        self.unk = 'UNK'
        self.ngram_freq = None
        self.ngram_probability = None
        self.smaller_ngram = None

    def getSentencesFromFile(self,filepath):
        '''
        INPUT - file in standard format
        OUTPUT - list of list of strings, each list of strings is a sentence
        '''
        file = open(filepath, 'r',encoding='utf-8')
        sentences = []
        for line in file:
            line = line.strip()
            sentence = line.split()
            for i in range(self.N-1):
                sentence.insert(0,'<S>')
                sentence.append('</S>')
            sentences.append(sentence)
        return sentences
    def saveRes(self, filepath,sentences, p, mesg):
        fo = open(f'{filepath}_res', 'w+')
        n = len(sentences)
        for i in range(n):
            fo.write(' '.join(sentences[i])+f' | PROB = {p[i]}\n')
        fo.close()
        
    def addKSmoothing(self, k):
        '''for each context add k to every word's frequecy'''
        (C,V) = self.ngram_freq.shape
        totals = list(map(lambda x: sum(x), self.ngram_freq))
        for i in range(C):
            self.ngram_probability[i]= (self.ngram_freq[i]+k)/(totals[i] + k*V)
    
    def goodTuringSmoothing(self):
        pass

    def kneserSmoothing(self):
        pass
    ######STUPID BACKOFF
    def stupidBackOffProbability(self, filepath):
        if(self.N>1 and self.smaller_ngrams==None):
            print("Smaller ngrams are not trained for backoff")
        sentences = self.getSentencesFromFile(filepath)
        probs = [self.stupidPofSentence(s) for s in sentences]
        self.saveRes(filepath, sentences, probs, 'Stupid backoff')
        return
        
    
    def stupidPofSentence(self, tokens):
        if(len(tokens)<self.N):
            return 0
        p = 1
        tn = len(tokens)
        for i in range(self.N-1, tn):
            context = '|'.join(tokens[i-self.N+1:i])
            p*= self.stupidPofWord(tokens[i], context)
        return p
    
    def stupidPofWord(self, word, context):
        p=0
        if((context in self.contexts) and (word in self.vocabulary)):
            p = self.ngram_probability[self.contexts[context]][self.vocabulary[word]]
        if(p==0 and self.smaller_ngram !=None):
            p = 0.4*self.smaller_ngram.stupidPofWord(word, context[1:])
        return p

    ##########INTERPOLATION
    def interpolationProbability(self, file_path, interpolation_params = []):
        if(len(interpolation_params)!=self.N):
            print('Number of interpolation params must be = ', self.N)
            return -1
        if(self.smaller_ngram == None):
            print('Smaller n gram is not trained while training')
            return -1
        sentences = self.getSentencesFromFile(file_path)
        probs = [self.interpolationPofSentence(s) for s in sentences]
        self.saveRes(file_path, sentences, probs, f'Interpolation {interpolation_params}')
        return

    def interpolationPofSentence(self,tokens,params):
        if(tokens == []):
            print('Empty tokens list')
            return
        p = 1
        tn = len(tokens)
        for i in range(self.N-1, tn):
            context = '|'.join(tokens[i-self.N+1:i])
            p*= self.interPofWord(tokens[i], context,params)
        return p
    
    def interPofWord(self, word, context, params):
        p=0
        if((context in self.contexts) and (word in self.vocabulary)):
            p = params[0]*self.ngram_probability[self.contexts[context]][self.vocabulary[word]]
        if(self.smaller_ngram !=None):
            p += self.smaller_ngram.interPofWord(word,context[1:], params[1:])
        elif(self.N > 1):
            print(f'Smaller n gram not present {self.N} interpolation')
        return p

    def train(self, filepath, train_smaller=False):
        t2 = None
        if(train_smaller and self.N>1):
            self.smaller_ngram = ngram(self.N-1)
            t2 = ThreadWithReturn(target=self.smaller_ngram.train, args=(filepath,train_smaller))
            t2.start()
        self.load(filepath)
        self.calculateProbability()
        if(train_smaller and self.N>1):
            t2.join()
        

    def load(self, filepath):
        print(f'setting frequency for {self.N}gram')
        #add vocabulary
        sentences = self.getSentencesFromFile(filepath)
        #get all the words
        #print(self.ngram_freq, self.ngram_freq.index)
        series = {}
        count=0
        for sentence in sentences:
            for word in sentence:
                if(word not in self.vocabulary):
                    self.vocabulary[word]=count
                    count+=1
        count = 0
        for sentence in sentences:
            sn = len(sentence)
            for i in range(self.N-1, sn):
                context = '|'.join(sentence[i-self.N+1:i])
                if(context not in self.contexts):
                    self.contexts[context] = count
                    count+=1
        self.ngram_freq = np.zeros((len(self.contexts), len(self.vocabulary)))
        for sentence in sentences:
            sn = len(sentence)
            for i in range(self.N-1, sn):
                context = '|'.join(sentence[i-self.N+1:i])
                self.ngram_freq[self.contexts[context]][self.vocabulary[sentence[i]]] +=1
        print(f'Contexts, Vocabulary size of {self.N}gram = ', self.ngram_freq.shape)

    
    def calculateProbability(self):
        start = time.time()
        print(f'Computing probability matrix of {self.N}gram')
        self.ngram_probability = np.zeros(self.ngram_freq.shape)
        (C,V) = self.ngram_freq.shape
        totals = []
        totals = list(map(lambda x: sum(x), self.ngram_freq))
        for i in range(C):
            self.ngram_probability[i] = self.ngram_freq[i]/totals[i]
        print(f'Probability computation of {self.N}gram complete, time taken = ',time.time()-start)
        #self.ngram_probability = self.ngram_freq/totals

    def getPofWord(self,word, context):
        if(word not in self.vocabulary):
            word = self.unk
        if(context not in self.contexts):
            return 0
        return self.ngram_probability[context][word]
    
    def getPofSentence(self, tokens):
        if(tokens == []):
            print('Empty tokens list')
            return
        p = 1
        tn = len(tokens)
        for i in range(self.N-1, tn):
            context = '|'.join(tokens[i-self.N+1:i])
            p*= self.getPofWord(tokens[i], context)
        return p

    def saveNgram(self):
        pickle.dump(self,open(f'{self.N}gram', 'wb'))
    


    


start = time.time()
model = ngram(5)
model.train('parsed.txt',True)
# model = pickle.load(open('3gram', 'rb'))
# model.saveNgram()