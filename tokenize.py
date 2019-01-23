#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:31:48 2019

@author: jeetu
"""

#importing nltk library 
import nltk
#importing sentence tokenizer and word tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

#sample paragraph
example_text="Hello Mr. Smith, how are you doing today? the weather is great and doing python is awesome. The sky is blue in colour."

#sentence tokenize
print(sent_tokenize(example_text))
#word tokenize
print(word_tokenize(example_text))

#output after tokenization
for i in sent_tokenize(example_text):
    print (i)
for i in word_tokenize(example_text):
    print(i)
    