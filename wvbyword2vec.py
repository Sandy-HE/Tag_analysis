# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:57:57 2020

@author: Sandy
"""

from gensim.models import Word2Vec
import pandas as pd

df = pd.read_csv("allprunedtags.csv", index_col=0)

#When I did pruning (1000,0.8, 0.0002), this leads to some tracks containing 0 tags.
#Remove such tracks, 504555 tracks is reduced to 470280 tracks.
df.dropna(subset = ["prunedtag"], inplace=True)

#build up list of lists of terms
temp2 = df['prunedtag'].apply(lambda x: x.split(";"))
terms_list = temp2.tolist()

#set up model
model = Word2Vec(terms_list,
                min_count=2,   # Ignore words that appear less than this
                size=4,      # Dimensionality of word embeddings
                workers=2,     # Number of processors (parallelisation)
                window=5,      # Context window for words during training
                iter=10)       # Number of epochs training over corpus
#sg = 1L

#word vectors
wordvec = model[model.wv.vocab]
wordvec_df = pd.DataFrame(data=wordvec)
wordvec_df['term'] = list(model.wv.vocab)

wordvec_df.to_csv('phrase_7685_cbow_D32.csv', index=False)