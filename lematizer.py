#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:57:37 2019

@author: jeetu
"""

from nltk.stem import WordNetLemmatizer

lematizer = WordNetLemmatizer()

print (lematizer.lemmatize('cats'))
print (lematizer.lemmatize('rocks'))
print (lematizer.lemmatize('cakes'))
print (lematizer.lemmatize('reading'))
print (lematizer.lemmatize('parks'))

#                       good better   because its adjective
print (lematizer.lemmatize('better', pos="a"))