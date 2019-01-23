#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:31:48 2019

@author: jeetu
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

example_text="Hello Mr. Smith, how are you doing today? the weather is great and doing python is awesome. The sky is blue in colour."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for i in sent_tokenize(example_text):
    print (i)

for i in word_tokenize(example_text):
    print(i)
    