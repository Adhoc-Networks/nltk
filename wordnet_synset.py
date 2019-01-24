#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:42:13 2019

@author: jeetu
"""

from nltk.corpus import wordnet

syns=wordnet.synsets("program")

#print(syns[0].examples())

synonyms=[]
antonyms=[]

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print (set(synonyms))
print (set(antonyms))

w1 =wordnet.synset("ship.n.01")
w2 =wordnet.synset("boat.n.01")

print (w1.wup_similarity(w2))

w1 =wordnet.synset("ship.n.01")
w2 =wordnet.synset("cat.n.01")

print (w1.wup_similarity(w2))

w1 =wordnet.synset("ship.n.01")
w2 =wordnet.synset("car.n.01")

print (w1.wup_similarity(w2))