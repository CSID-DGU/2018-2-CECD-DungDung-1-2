#!/usr/bin/env python
# coding: utf-8

# ## 준비된 vocab, tokenizer, kor2vec, Classifier를 사용하여 입력된 문장을 분류하는 코드

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as D


# In[2]:


from kor2vec import Kor2Vec


# In[3]:


from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer
from soynlp.hangle import jamo_levenshtein
import pickle


# In[4]:


import re


# In[5]:


SEQ_LEN = 10
# 받을때는 텍스트 파일에 있는 문장을 읽어오자


# In[6]:


class SentenceClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size):
        super(SentenceClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, BATCH_SIZE, self.hidden_dim),
               torch.zeros(1, BATCH_SIZE, self.hidden_dim))
    
    # x = embedding.vectorizeSentence(list of sentence)
    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out[:,9,:]
        y = self.hidden2label(lstm_out)
        
        # y = self.hidden2label(lstm_out, -1)
        result = F.log_softmax(y, dim=1)
        
        return result


# In[7]:


class LabelMaker:
    def __init__(self, kor2vecFileName = "./nlp/embedding.model", tokenizerFileName = "./nlp/tokenizer.pkl", 
                 calssifierFileName = "./nlp/classifier.model", vocabFileName = "./nlp/vocab.txt", seq_len = SEQ_LEN):
        self.setKor2Vec(kor2vecFileName)
        self.setTokenizer(tokenizerFileName)
        self.setClassifier(calssifierFileName)
        self.setVocab(vocabFileName)
        
        self.seq_len =seq_len
        
    def setKor2Vec(self, kor2vecFileName):
        self.kor2vec = Kor2Vec.load(kor2vecFileName)
        
    def setTokenizer(self, tokenizerFileName):
        with open(tokenizerFileName,'rb') as f:
            self.tokenizer = pickle.load(f)
    
    def setClassifier(self, calssifierFileName):
        with open(calssifierFileName,'rb') as f:
            self.classifier = pickle.load(f)
        
    def setVocab(self, vocabFileName):
        self.vocab = []
        f = open(vocabFileName, 'r')
        
        while True:
            word = f.readline()
            if not word: 
                break
            else :
                self.vocab.append(word[:-1])
        f.close()        
        
    def tokenizeSentence(self, sentence): 
        sentence = sentence.replace(" ", "")
        result = self.tokenizer.tokenize(sentence)
        return result
    
    def checkOOV(self, words):
        new_words = []
        for w in words:
            if w in self.vocab:
                new_words.append(w)
            else:
                baseline = 0.7
                new_w = ""
                for v in self.vocab:
                    distance = jamo_levenshtein(v, w)
                    if distance <= baseline:
                        baseline = distance
                        new_w = v
                # 유사한 단어가 있을 때
                if new_w != "" and baseline <= 0.7:
                    new_words.append(new_w)
                # 유사한 단어가 없을 때
                else:
                    new_words.append(w)
        # print(new_words)
        return new_words

    def deleteSymbol(self, sentence):
        f = re.compile('[^ ㄱ-ㅣ가-힣|A-Z|a-z|0-9 ]+') 
        result = f.sub('', sentence)
        
        return result
    
    def onlyKorean(self, sentence):
        f = re.compile('[^ ㄱ-ㅣ가-힣 ]+') 
        result = f.sub('', sentence)
        
        return result
    
    def vectorizeSentence(self, sentence):
        x = self.kor2vec.to_seqs(sentence, seq_len = self.seq_len)
        x = self.kor2vec(x)
        
        return x
        
    def modelForward(self, vectors):
        self.classifier.hidden = (torch.zeros(1, 1, self.classifier.hidden_dim), torch.zeros(1, 1, self.classifier.hidden_dim))
        result = self.classifier.forward(vectors)
        _, result = torch.max(result, 1)
        
        return result
    
    # 원문장에서 기호 삭제한 문장, 원문장에서 기호 및 외국어 삭제한 문장, 레이블
    def classifySentence(self, sentence):    
        noSymbol_sentence = self.deleteSymbol(sentence)
        
        korean_sentence = self.onlyKorean(sentence)
        words = self.tokenizeSentence(korean_sentence)
        words = self.checkOOV(words)
        fixed_sentence = " ".join(words)
        vectors = self.vectorizeSentence([fixed_sentence])
        
        """
        original_words = noSymbol.split(" ") 
        original_words = self.checkOOV(original_words)
        oovChecked = " ".join(original_words)
        """
        
        result = self.modelForward(vectors)
        
        return noSymbol_sentence, fixed_sentence, result


# In[8]:


lm = LabelMaker()


# In[9]:


# print(lm.classifySentence("김동환 교수님은 어떤거 연구하세요?"))

f = open("question.txt", 'r')
user_question = f.readline()
f.close()


# In[10]:


result_label = lm.classifySentence(user_question)


# In[11]:


f = open("label.txt", 'w')
f.write(str(result_label[2].item()))
f.close()

