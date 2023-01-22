
class unigram:
    def __init__(self):
        self.unigram_freq = {}
        self.unigram_probability = {}
    def train_unigram(sentences):
        unigram_freq = {}
        for sentence in sentences:
            for word in sentence:
                if(word in unigram_freq):
                    unigram_freq[word]+=1
                else:
                    unigram_freq[word]=1
    def smoothing(self):
        pass
    def get_unigram_probability(self,sentence_list):
        p = 1
        for word in sentence_list:
            P = p*self.unigram_probability[word]
        return p

class bigram:
    pass

class trigram:
    pass

class four_gram:
    pass

class five_gram:
    pass

    