import codecs
import pandas as pd
import string
# from keras.preprocessing import text, sequence

# load data into the memory
train = pd.read_csv('./data/train.tsv', sep='\t')
# print(train.head())
test = pd.read_csv('./data/test.tsv', sep='\t')
# print(test.head())


# define apply function to clean the data
def process_text(phrase):
    phrase = phrase.lower()
    punctuation = string.punctuation
    for punc in punctuation:
        phrase = phrase.replace(punc, ' {} '.format(punc))
    return phrase


x_train = train['Phrase'].apply(process_text)
x_test = test['Phrase'].apply(process_text)
y_train = train['Sentiment']

# extrct the corpus and save it to the disk
corpus = '\n'.join(list(x_train)+list(x_test))
f = codecs.open('./data/corpus.txt', 'w')
f.write(corpus)
f.close()
# tokenizer = text.Tokenizer()
# tokenizer.fit_on_texts(list(x_train) + list(x_test))
