# import modules & set up logging
import logging
import os
from gensim.models import word2vec
'''print log'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
'''feed the text'''
sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt')
'''trainning'''
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=5,size=100)
print(model['侯亮平'])
print(model)