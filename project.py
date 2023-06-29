# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    response = requests.get(url)
    string = response.text
    firstidx = string.find(' ***') + 4
    secondidx = string.find('*** END')
    new = string[firstidx: secondidx].replace('\r\n', '\n')
    time.sleep(5)
    return new


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    new = re.sub('\n(\n)+', '\x02', book_string, count = 1)
    new = new[::-1]
    new2 = re.sub('\n(\n)+', '\x03', new, count = 1)
    new2 = new2[::-1]
    new3 = re.sub('\n(\n)+', '\x03 \x02', new2)
    new4 = re.findall(r'(\w+\b|[^a-zA-Z0-9_ ])', new3)
    return new4

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)

    def train(self, tokens):
        idx = pd.Series(tokens).unique()
        dta = np.array([1] * len(idx)) / len(idx)
        self.ser = pd.Series(data = dta, index = idx)
        return self.ser

    def probability(self, words):
        if all(pd.Series(words).isin(self.ser.index)):
            return self.ser.values[0] ** len(words)
        else:
            return 0

    def sample(self, M):
        sent = np.random.choice(self.ser.index, size = M, p =self.ser.values)
        return ' '.join(sent)

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):

    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        self.tokens = tokens
    def train(self, tokens):
        self.ser = pd.Series(tokens).value_counts() / len(tokens)
        return self.ser

    def probability(self, words):
        #change this
        def change(num):
            return ser1[num]
        if all(pd.Series(words).isin(self.ser.index)):
            return pd.Series(words).apply(change).prod()
        else:
            return 0

    def sample(self, M):
        sent = np.random.choice(self.ser.index, size = M, p = self.ser.values)
        return ' '.join(sent)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):

    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!

        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        lst = []
        for i in (range(len(tokens) - self.N + 1)):
            lst.append(tuple(tokens[i: i + self.N]))
        return lst

    def train(self, ngrams):
        def tup_split(tup):
            return tup[0:self.N - 1]

        df = pd.DataFrame()
        df['ngram'] = pd.Series(ngrams)
        df['n1gram'] = df['ngram'].apply(tup_split)

        n1gram_counts = df['n1gram'].value_counts().reset_index()
        n1gram_counts.columns = ['n1gram', 'p1']

        ngram_counts = df['ngram'].value_counts().reset_index()
        ngram_counts.columns = ['ngram', 'p2']

        df = pd.merge(df, n1gram_counts, left_on='n1gram', right_on='n1gram')
        df = pd.merge(df, ngram_counts, left_on='ngram', right_on='ngram')

        df['prob'] = df['p2'] / df['p1']
        df = df.drop(columns=['p1', 'p2'])

        df = df.drop_duplicates()
        return df


    def probability(self, words):
        ngrams = self.create_ngrams(words)
        def change(num):
            return self.mdl[self.mdl['ngram'].astype(str) == str(num)]['prob']
        if all(pd.Series(ngrams).isin(self.mdl['ngram'])):
            #regular prob
            prob = pd.Series(ngrams).apply(change).prod().prod()
            #shld loop
            count = self.N
            prev = self.prev_mdl
            while count > 2:
                prob = prob * prev.mdl[self.prev_mdl.mdl['ngram'] == prev.ngrams[0]]['prob'][0]
                count -= 1
                prev = prev.prev_mdl
            #unigram model
            prob = prob * prev.ser[words[0]]
            return prob
        else:
            return 0


    def sample(self, M):
        def sample_next(prev_ngram):
            candidates = self.mdl.loc[self.mdl['n1gram'].astype(str) == str(prev_ngram)]
            prob = candidates['prob'].values
            next_word = np.random.choice(candidates['ngram'], p=prob)
            return next_word[-1]

        def generate_tokens(current_ngram, length):
            if length == 0:
                return []
            else:
                next_word = sample_next(current_ngram)
                return [next_word] + generate_tokens(current_ngram[1:] + (next_word,), length - 1)

        output = ['\x02']
        output.extend(generate_tokens(tuple(['\x02'] * (self.N - 1)), M - 1))
        output.append('\x03')
        return ' '.join(output)
