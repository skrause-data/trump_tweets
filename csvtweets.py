#!/usr/bin/env python
# coding: utf-8
'''
Step 1) Prepare tweets from csv file

Tweets from June 15, 2015 (announcement election) to May 28, 2020
source: http://www.trumptwitterarchive.com/archive
'''

import pandas as pd
import string

from nltk import word_tokenize
from collections import Counter


def csvimport(csv):
    """Import local csv file and prepare it for further analyses."""
    df_raw = pd.read_csv(csv, sep='\t', encoding = "utf-8")
    df = df_raw[~df_raw["text"].str.contains("RT @", na=True)]          #remove retweets
    df['text'] = df['text'] \
    .replace(r'http\S+', '', regex=True) \
    .replace(r'www\S+', '', regex=True) \
    .replace(r'&amp;', '&', regex=True) \
    .replace(r'&gt;', '>', regex=True) \
    .replace(r'&lt;', '<', regex=True) \
    .replace(r'@realDonaldTrump', '', regex=True) \
    .replace(r'U.S.', 'UnitedStates', regex=True) \
    .replace(r'u.s.', 'UnitedStates', regex=True) \
    .replace(r'United States', 'UnitedStates', regex=True) \
    .replace(r'united states', 'UnitedStates', regex=True)   #remove html clutter & replace U.S.-> UnitedStates
    df['text'] = df['text'].str.strip()
    df['text'].replace("", float("NaN"), inplace=True)
    df = df.dropna()
    return df


def stop(text, stoplist):
    """Remove stopwords, punctuation, and convert words to lowercase."""
    text = text.str.lower()
    pat = r'\b(?:{})\b'.format('|'.join(stoplist))
    text = text.replace(pat, '', regex=True)
    text = text.str.replace('[{}]'.format(string.punctuation), '')
    return text




