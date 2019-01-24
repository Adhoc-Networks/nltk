#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:02:49 2019

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
            chunkGram = r"""Chunk:{<.*>+}
                                    }<VB.?|IN|DT|TO>+ {"""
                                    #opposite bracket for excluding from the list/chunk
            chunkParser =nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #output in the tree format
            chunked.draw()




    except Exception as e:
        print(str(e)) 
        
process_content() 