#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:52:50 2019

@author: jeetu
"""

import nltk
#importing the speech of george bush from state union
from nltk.corpus import state_union
# PunktSentenceTokenizer is unsupervised ML
from nltk.tokenize import PunktSentenceTokenizer

#training the data of 2005 speech
train_text = state_union.raw("2005-GWBush.txt")
#sample text of 2006
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer=PunktSentenceTokenizer(train_text)
#tokenized by sentenced
tokenized = custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            #RegularExpression taking adverb verb noun proper and noun
            chunkGram = r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?} """
            
            chunkParser =nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #output in the tree format
            chunked.draw()




    except Exception as e:
        print(str(e)) 
        
process_content() 