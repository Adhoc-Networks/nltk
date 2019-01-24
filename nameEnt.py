#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:50:16 2019

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
        for i in tokenized[5:]:
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            
            nameEnt = nltk.ne_chunk(tagged)
            
            nameEnt.draw()





    except Exception as e:
        print(str(e)) 
        
process_content() 

'''
NE Type	    Examples
ORGANIZATION	   Georgia-Pacific Corp., WHO
PERSON	     Eddy Bonte, President Obama
LOCATION	   Murray River, Mount Everest
DATE	   June, 2008-06-29
TIME	   two fifty a m, 1:30 p.m.
MONEY	  175 million Canadian Dollars, GBP 10.40
PERCENT     	twenty pct, 18.75 %
FACILITY	   Washington Monument, Stonehenge
GPE	    South East Asia, Midlothian
'''