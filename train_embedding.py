from gensim.models import Word2Vec, FastText
import codecs
import gc
import time
import os
f = codecs.open('./data/corpus.txt', 'r')
corpus = f.readlines()
num = len(corpus)
for i in range(num):
    corpus[i] = corpus[i].split()

os.makedirs('./vectors', exist_ok= True)

start = time.time()
print('lauch the Word2Vec trainning process')
model = Word2Vec(corpus, size=300, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format('./vectors/Word2Vec.txt', binary=False)
del model
gc.collect()
end = time.time()
duration = end - start
print('Word2Vec trainning and model saving cost {:.2f} seconds'.format(duration))
print('lauch the FastText trainning process')
start = time.time()
model = FastText(corpus, size=300, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format('./vectors/FastText.txt', binary=False)
end = time.time()
duration = end - start
print('FastText trainning and model saving cost {:.2f} seconds'.format(duration))
del model
gc.collect()
