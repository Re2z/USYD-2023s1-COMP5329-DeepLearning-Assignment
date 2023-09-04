import numpy as np
import pandas as pd
from io import StringIO
import re
import nltk
import gensim.downloader
from collections import Counter

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def caption_extract(caption):  # extract the main word in caption
    caption_in_word = []
    for sentences in caption:
        sentence_in_word = []
        sentences = sentences.lower()  # lowercase
        sentences = re.sub(r'[^A-Za-z]+', ' ', sentences)  # remove all symbols and digits
        for word in sentences.split():
            if word not in set(stopwords.words('english')):  # remove stopword
                sentence_in_word.append(word)

        lemma_word = [WordNetLemmatizer().lemmatize(item) for item in sentence_in_word]  # lammatisation

        # token = [word_tokenize(word) for word in text_le]
        caption_in_word.append(lemma_word)
    return caption_in_word


with open("../../train.csv") as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    train_caption = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    train_caption = train_caption.drop(columns='Caption').join(train_caption['Caption'].str.replace('\"', ''))
with open("../../test.csv") as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    test_caption = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    test_caption = test_caption.drop(columns='Caption').join(test_caption['Caption'].str.replace('\"', ''))

whole_caption = pd.concat([train_caption['Caption'], test_caption['Caption']], axis=0, ignore_index=True)
caption_processed = caption_extract(whole_caption)
max_seq_len = max(len(s) for s in caption_processed)

# Download the Word2Vec model
embedding_model = gensim.downloader.load('word2vec-google-news-300')
# vector size of each word
embedding_dim = embedding_model.vector_size

# Add the word appeared in the caption extraction to the word list
word_list = []
for sentence in caption_processed:
    for word in sentence:
        word_list.append(word)

# Select the word with more than 5 appearance in the caption to the vocabulary set
vocab_set = set()
counter = Counter(word_list)
for i in counter:
    if counter[i] >= 5:
        vocab_set.add(i)

# Add the PAD and UNKNOWN to the vocabulary set, each word in the vocabulary list only exist once
vocab_set.add('[PAD]')
vocab_set.add('[UNKOWN]')
vocab_list = list(vocab_set)
vocab_list.sort()

# word dictionary
vocab_dic = {}
emb_table = []

# Creating the word model based on the vocabulary
for i, word in enumerate(vocab_list):
    vocab_dic[word] = i
    if word in embedding_model:
        emb_table.append(embedding_model[word])
    else:
        emb_table.append([0] * embedding_dim)
emb_table = np.array(emb_table)


def tokenizer(caption):
    tokenize = []
    caption = caption_extract(caption)
    for item in caption:
        temp = [vocab_dic[word] if word in vocab_dic else vocab_dic['[UNKOWN]'] for word in item]
        if len(temp) < max_seq_len:
            temp += [vocab_dic['[PAD]']] * (max_seq_len - len(temp))
        else:
            temp = temp[:max_seq_len]
        tokenize.append(temp)
    return tokenize
