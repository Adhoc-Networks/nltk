#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:10:26 2019

@author: jeetu
"""

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample= gutenberg.raw("bible-kjv.txt")

sent_token= sent_tokenize(sample)

print (sent_token[:15]) 