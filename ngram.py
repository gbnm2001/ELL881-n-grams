import numpy as np
from threading import Thread
import pickle
import time
import math
import random
import sys
import os
#from plot_perp import *
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
        self.vocabulary = {'UNK':0}
        self.contexts = dict()
        self.unk = 'UNK'
        self.ngram_freq = None
        self.ngram_probability = None
        self.smaller_ngram = None
        self.continuation = None
        self.vocab_limit = 20000
        self.freq_sum=[]
        self.continuation_sum = 0
    def getSentencesFromFile(self,filepath):
        '''
        INPUT - file in standard format
        OUTPUT - list of list of strings, each list of strings is a sentence
        '''
        file = open(filepath, 'r',encoding='utf8')
        sentences = []
        for line in file:
            line = line.strip()
            sentence = line.split()
            for i in range(self.N-1):
                sentence.insert(0,'<S>')
                sentence.append('</S>')
            sentences.append(sentence)
        return sentences
    def saveRes(self, filepath,sentences, p, mesg=''):
        perp = []
        for i in range(len(p)):
            den = len(sentences[i])-2*(self.N-1)
            if(den == 0 or p[i]==0):
                perp.append(float('inf'))
            else:
                perp.append((1/p[i])**(1/den))
        
        filepath = filepath.split('.')[0]
        fo = open(f'{filepath}_{mesg}_res.txt', 'w+',encoding='utf8')
        n = len(sentences)
        fo.write(mesg+'\n')
        for i in range(n):
            s = ' '.join(sentences[i])
            fo.write(s+f' PROB={p[i]} PERP={perp[i]}\n')
        print(f'Result saved in {filepath}_{mesg}_res.txt')
        fo.close()
        #plot(f'{filepath}_res.txt',mesg)
    def saveNgram(self):
        pickle.dump(self,open(f'{self.N}gram', 'wb'))
    
#####LOADING FUNCTIONS
    def train(self, filepath, smallerNgram = None):
        self.load(filepath)
        self.calculateProbability()
        self.smaller_ngram = smallerNgram
        

    def load(self, filepath):
        print(f'setting frequency for {self.N}gram')
        #add vocabulary
        sentences = self.getSentencesFromFile(filepath)
        #get all the words
        #print(self.ngram_freq, self.ngram_freq.index)
        series = {}
        #####ASSIGN INDEX TO VOCABULARY WORDS
        vcount = {}
        index=len(self.vocabulary)#0th is 'UNK'
        for sentence in sentences:
            for word in sentence:
                if(word in vcount):
                    vcount[word]+=1
                else:
                    vcount[word] = 1
        #keep only top 20,000 words by frequency in corpus DANGER WE DONT KNOW WHAT IS REPLACED
        self.vocabulary = dict(sorted(list(vcount.items()), key=lambda x: -1*x)[:self.vocab_limit])
        self.vocabulary[self.unk] = 0
        index = 0
        for i in self.vocabulary:
            self.vocabulary[i] = index
            index+=1
        
        index = len(self.contexts)
        for sentence in sentences:
            sn = len(sentence)
            for i in range(self.N-1, sn):
                context = '|'.join(sentence[i-self.N+1:i])
                if(context not in self.contexts):
                    self.contexts[context] = index
                    index+=1
        self.ngram_freq = np.zeros((len(self.contexts), len(self.vocabulary)), dtype=np.uint16)
        for sentence in sentences:
            sn = len(sentence)
            for i in range(self.N-1, sn):
                context = '|'.join(sentence[i-self.N+1:i])
                if(sentence[i] in self.vocabulary):
                    self.ngram_freq[self.contexts[context]][self.vocabulary[sentence[i]]] +=1
                else:
                    self.ngram_freq[self.contexts[context]][self.vocabulary[self.unk]] +=1
        
        print(f'Contexts, Vocabulary size of {self.N}gram = ', self.ngram_freq.shape)

    
    def calculateProbability(self):
        start = time.time()
        self.ngram_probability = np.zeros(self.ngram_freq.shape, dtype=np.float16)
        print(f'Computing probability matrix of {self.N}gram')
        (C,V) = self.ngram_freq.shape
        totals = []
        totals = list(map(lambda x: sum(x), self.ngram_freq))
        for i in range(C):
            self.ngram_probability[i] = self.ngram_freq[i]/totals[i]
        print(f'Probability computation of {self.N}gram complete, time taken = ',time.time()-start)
        #self.ngram_probability = self.ngram_freq/totals

#########simple ngram functions
    def getPofWord(self,word, context):
        if(word not in self.vocabulary):
            word = self.unk
        if(context not in self.contexts):
            return 0
        return self.ngram_probability[self.contexts[context]][self.vocabulary[word]]
    
    def getPofSentence(self, tokens):
        if(tokens == []):
            return
        p = 1
        tn = len(tokens)
        for i in range(self.N-1, tn):
            context = '|'.join(tokens[i-self.N+1:i])
            p*= self.getPofWord(tokens[i], context)
        return p

    def simpleProbability(self, file_path,mesg='Simple'):
        sentences = self.getSentencesFromFile(file_path)
        probs = [self.getPofSentence(s) for s in sentences]
        self.saveRes(file_path, sentences, probs, mesg+str(self.N))
    

########ADD K SMOOTHING
    def addKSmoothing(self, k):
        '''for each context add k to every word's frequecy'''
        self.ngram_probability = np.zeros(self.ngram_freq.shape, dtype=np.float16)
        (C,V) = self.ngram_freq.shape
        totals = list(map(lambda x: sum(x), self.ngram_freq))
        for i in range(C):
            self.ngram_probability[i]= (self.ngram_freq[i]+k)/(totals[i] + k*V)
    def add1Smoothing(self):
        self.addKSmoothing(1)
    

#########GOOD TURING
    def goodTuringSmoothing(self):
        print(f'Performing good Turing smoothing of {self.N}gram')
        (C,V) = self.ngram_freq.shape
        for i in range(C):
            print(i,end = ' ')
            unique, counts = np.unique(self.ngram_freq[i], return_counts=True)
            dc = dict(zip(unique,counts))
            if(1 in dc):
                self.ngram_probability[self.ngram_freq==0] = dc[1]/sum(self.ngram_freq[i])
            for j in range(V):
                c = self.ngram_freq[i][j]
                if ((c+1) in dc):
                    self.ngram_probability[i][j] = (c+1)*dc[c+1]/dc[c]*self.ngram_probability[i][j]

########Kneser
    def calculateContinuationCount(self):
        #non_zeros = np.count_nonzero(self.ngram_freq)
        print(f'Calculating continuation count for {self.N}gram')
        def f(a):
            return np.count_nonzero(a)
        self.continuation = np.apply_along_axis(f, 0, self.ngram_freq)

    def kneserNeyPofWord(self, word, context, higher):
        if(word not in self.vocabulary):
            word = self.unk
        a= 0
        if(higher and (context in self.contexts)):
            a = max((self.ngram_freq[self.contexts[context]][self.vocabulary[word]] - 0.75)/self.freq_sum[context] , 0)
        else:
            a = max(self.continuation[self.vocabulary[word]]-0.75,0)/self.continuation_sum
        lamda = 0.4
        b=0
        if(self.smaller_ngram!=None):
            b = self.smaller_ngram.kneserNeyPofWord(word, context,False)
        elif(self.N >1):
            print('Smaller ngram not present for ', self.N,'gram')
        return a+lamda*b

    def kneserNeyPofSentence(self, tokens):
        if(len(tokens)<self.N):
            return 0
        p = 1
        tn = len(tokens)
        for i in range(self.N-1, tn):
            context = '|'.join(tokens[i-self.N+1:i])
            p*= self.kneserNeyPofWord(tokens[i], context)
        return p
    def kneserNeyProbability(self, filepath):
        if(self.N>1 and self.smaller_ngram==None):
            print("Smaller ngrams are not trained for backoff")
        sentences = self.getSentencesFromFile(filepath)
        self.continuation_sum = sum(self.continuation)
        self.freq_sum = np.apply_along_axis(sum, 1, self.ngram_freq)

        probs = [self.kneserNeyPofSentence(s) for s in sentences]
        self.saveRes(filepath, sentences, probs, f'Kneser{self.N}')
        return
#####STUPID BACKOFF
    def stupidBackOffProbability(self, filepath):
        if(self.N>1 and self.smaller_ngram==None):
            print("Smaller ngrams are not trained for backoff")
        sentences = self.getSentencesFromFile(filepath)
        probs = [self.stupidPofSentence(s) for s in sentences]
        self.saveRes(filepath, sentences, probs, f'Stupid backoff{self.N}')
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
        if(word not in self.vocabulary):
            word = self.unk
        p=0
        if(context in self.contexts):
            p = self.ngram_probability[self.contexts[context]][self.vocabulary[word]]
        if(p==0 and self.smaller_ngram !=None):
           p = 0.4*self.smaller_ngram.stupidPofWord(word, context[1:])
        elif(p==0 and self.N>1):
           print(f'Ngram not present at {self.N}gram')
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
        self.saveRes(file_path, sentences, probs, f'Interpolation{self.N}')
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
        if(word not in self.vocabulary):
            word = self.unk
        p=0
        if(context in self.contexts):
            p = params[0]*self.ngram_probability[self.contexts[context]][self.vocabulary[word]]
        if(self.smaller_ngram !=None):
            p += self.smaller_ngram.interPofWord(word,context[1:], params[1:])
        elif(self.N > 1):
            print(f'Smaller n gram not present {self.N} interpolation')
        return p

########SAMPLING FUNCTIONS
    def sampleWord(self, context):
        if(context not in self.contexts):
            return -1
        x = random.random()
        acc = 0
        i=-1
        V = len(self.vocabulary)
        while(i<V-1 and acc<x):
            i+=1
            acc+=self.ngram_probability[self.contexts[context]][i]
        for (k,v) in self.vocabulary.items():
            if(v==i):
                return k
            
    def sampleSentence(self, length):
        context = ['<S>']*(self.N-1)
        context = '|'.join(context)
        i=0
        res = []
        while(i<length):
            w = self.sampleWord(context)
            if(w!=-1):
                i+=1
                if(self.N>1):
                    context = context.split('|')
                    context.append(w)
                    context = context[1:]
                    context = '|'.join(context)
                res.append(w)
            else:
                break
        return res



def testSimpleNgram(n,filepath, testFile):
    print(f'Testing simple {n}gram')
    model = ngram(n)
    model.train(filepath)
    s = model.simpleProbability(testFile,f'Simple {n}gram ')
    f = open(f's{n}samples.txt','w+')
    for i in range(100):
        f.write(' '.join(model.sampleSentence(100))+'\n')
    
def testAddOne(n,filepath, testFile):
    print(f'add one {n}gram')
    model = ngram(n)
    model.load(filepath)
    model.add1Smoothing()
    s = model.simpleProbability(testFile, f'add one {n}gram ')
    f = open(f'ao{n}samples.txt','w+')
    for i in range(100):
        f.write(' '.join(model.sampleSentence(100))+'\n')

def testGoodTuring(n,filepath, testFile):
    print(f'Good turing {n}gram')
    model = ngram(n)
    model.train(filepath)
    model.goodTuringSmoothing()
    s = model.simpleProbability(testFile,f'Good turing {n}gram')
    f = open(f'goodturing{n}samples.txt','w+')
    for i in range(100):
        f.write(' '.join(model.sampleSentence(100))+'\n')


####The following require smaller ngrams also
def trainModel(models, i, filepath):
    models[i].train(filepath)

def testBackOff(n,filepath, testFile):
    print(f'Backoff {n}gram')
    models = [ngram(i) for i in range(1,n+1)]
    models.insert(0,None)
    for i in range(1,len(models)):
        models[i].train(filepath)
    for i in range(1,n+1):
        models[i].smaller_ngram = models[i-1]
    s = models[-1].stupidBackOffProbability(testFile)


def testInterpolation(n,filepath, testFile, params = [1,0.4,0.16,0.0216,0.01]):
    print(f'Interpolation {n}gram')
    print(f'Lambda parameters = {params}')
    models = [ngram(i) for i in range(1,n+1)]
    models.insert(0,None)
    for i in range(i,len(models)):
        models[i].train(filepath)
    for i in range(1,len(models)):
        models[i].smaller_ngram = models[i-1]
    s = models[-1].interpolationProbability(testFile,params)

def calculateContinuation(models,i,filepath):
    models[i].load(filepath)
    models[i].calculateContinuationCount()

def testKneserNey(n, filepath,testfile):
    print(f'Kneser Ney {n}gram')
    models = [ngram(i) for i in range(1,n+1)]
    models.insert(0,None)
    for i in range(i,len(models)-1):
        models[i].load(filepath)
        models[i].calculateContinuationCount()
    models[-1].train(filepath)
    for i in range(1,n+1):
        models[i].smaller_ngram = models[i-1]
    s = models[-1].kneserNeyProbability(testfile)



def printUsage():
    x = ''
    if(os.name == 'posix'):
        x = '3'
    print(f'USAGE: python{x} ngram.py ngram_type N training_file_path test_file_path')
    
    print('''ngram types - simple, addone, goodturing, backoff, interpolation, kneserney\nN > 0''')
if(len(sys.argv)==5):
    if(sys.argv[1] == 'simple'):
        testSimpleNgram(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif(sys.argv[1]== 'addone'):
        testAddOne(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif(sys.argv[1]== 'goodturing'):
        testGoodTuring(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif(sys.argv[1]== 'backoff'):
        testBackOff(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif(sys.argv[1]== 'interpolation'):
        testInterpolation(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif(sys.argv[1]== 'kneserney'):
        testKneserNey(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        print('INCORRECT NGRAM TYPE')
        printUsage()
else:
    printUsage()

