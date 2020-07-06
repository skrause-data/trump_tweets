#!/usr/bin/env python
# coding: utf-8
"""The project aims at sentiment analyzes of Trump Tweets:

    a) Valence (value: positive vs. negative vs. neutral)
    b) Basic Emotions Plutchik's (wheel of emotion)

Moreover, both valence and basic emotions are correlated with stock market values.
    

Used packages:
pip install nltk
pip install pandas
pip install wordcloud 
pip install vaderSentiment
pip install tweepy
pip install plotly
pip install NRCLex
"""

import os
import string
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from plotly.offline import plot
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nrclex import NRCLex

#Own moduls
import csvtweets



####################
# CSV Data Import
####################

df = csvtweets.csvimport("trump_2015.csv")  #import Trump tweets (local csv file) into a pd df
stoplist = stopwords.words('english')   #List of English stopwords (e.g., 'the', 'a', 'we')
stop = csvtweets.stop(df.text, stoplist)  #remove stopwords, punctuation and lowercase conversion
df = pd.concat([df,stop.apply(pd.Series)],1)
df = df.rename(columns={0: 'text w/o stopwords'})

# Word count & plot
word_counts = Counter(word_tokenize('\n'.join(df['text w/o stopwords'])))
word_counts.most_common(50)

words = []
for i in range (0, 10):
    words.append(word_counts.most_common(10)[i][0])

rank = []
for i in range (0, 10):
    rank.append(word_counts.most_common(10)[i][1])


# Frequencies table of word count
headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

fig = go.Figure(data=[go.Table(
    header = dict(
      values = ['Frequencies', 'Word'],
      line_color = 'darkslategray',
      fill_color = headerColor,
      align = ['center','center'],
      font = dict(color='white', size = 22), 
      height = 40
    ),
    cells=dict(
      values= [rank, words],
      line_color='darkslategray',
      fill_color = [[rowOddColor,rowEvenColor]*10],
      align = ['center', 'center'],
      font = dict(color = 'darkslategray', size = 16),
      height = 25)
      )
])
fig.update_layout(width=450, height=800)
plot(fig)
fig.write_image("01-word_count_freq.png")


#Plot Wordcloud
cloudtext = pd.Series.to_string(df['text w/o stopwords'])
cloudtext = cloudtext\
.replace('unitedstates', 'United States')\
.replace('trump', 'Trump')\
.replace('donald', 'Donald')\
.replace('foxnew', 'Fox News')\
.replace('fake new', 'fake news')\
.replace('Newss', 'News')\
.replace('hillary', 'Hillary')\
.replace('clinton', 'Clinton')\
.replace('newss', 'news')\
.replace('america', 'America')\
.replace('american', 'American')\
.replace('china', 'China')

mask = np.array(Image.open(r'background.png'))

wc = WordCloud(
    max_font_size=300, 
    min_font_size=8, 
    max_words=150, 
    mask=mask, 
    background_color='red',
    contour_width=2, 
    contour_color='black')\
.generate(cloudtext)

image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,15])
plt.imshow(wc.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")
plt.show()

wc.to_file(r'wordcloud.png')


# Tokenizing and stemming 
tokenizer = nltk.RegexpTokenizer(r"\w+")
ps = PorterStemmer()

df['token'] = df['text w/o stopwords'].apply(lambda x :filter(None,tokenizer.tokenize(x)))
df['stem']=df['token'].apply(lambda x : [ps.stem(y) for y in x])
df['stemmed_sentence']=df['stem'].apply(lambda x : " ".join(x))
#df.to_excel("df.xlsx") 


#Filter Tweets for stock market topics
stockfilter = ['trade', 'inflat', 'economi', 'growth', 'manipul', 'currenc', 'dollar', 'china', 'fed', 'powel', 'xi', 'tariff', 'impeach']

df["filter"] = df['stemmed_sentence'].apply(lambda x: 1 if any(i in x for i in stockfilter) else 0)

df_stocktweets = df[df['filter'] == 1]
df_stocktweets = df_stocktweets.drop(columns=['token', 'filter'])


####################
# Sentiment analysis
####################

######
# Valance using VADER

# VADER (Valence Aware Dictionary and sEntiment Reasoner)
# see: https://github.com/cjhutto/vaderSentiment and
#      http://www.nltk.org/_modules/nltk/sentiment/vader.html

# VADER labels:  positive, negative, neutral and a compound value -1 to +1 (normalizied sum of all 3 values)

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

sentiment = df_stocktweets['text'].apply(lambda x: analyzer.polarity_scores(x))
df_stocktweets = pd.concat([df_stocktweets,sentiment.apply(pd.Series)],1)  # Based on the VADER docs using the original text data, instead of processed text data, is recommended.


######
# Basic emotions using NRC Word-Emotion Association Lexicon 

# see:  https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
#       https://pypi.org/project/NRCLex/ und v.a.
#       http://saifmohammad.com/ 

# Why are basic emotions so important?
# see:  https://en.wikipedia.org/wiki/Robert_Plutchik und 
#       https://en.wikipedia.org/wiki/Discrete_emotion_theory


def emo(str):
    """affect_frequencies returns a value between 0 and 1 for all emotions per word."""
    text_object = NRCLex(str)
    return text_object.affect_frequencies

emotions = df_stocktweets['stemmed_sentence'].apply(lambda x: emo(x))
df_stocktweets = pd.concat([df_stocktweets,emotions.apply(pd.Series)],1)  # Using stemmed words are recommended by the NRCLex emotion libary.


######
# Combine sentiment and basic emotion values per day 

df_stocktweets['created_at'] = pd.to_datetime(df_stocktweets['created_at'])
df_sentiment = df_stocktweets.groupby([df_stocktweets['created_at'].dt.date])['compound', 'neg', 'neu', 'pos', 'fear', 'anger', 'anticip', 'trust', 'surprise', 'sadness', 'disgust', 'joy', 'positive', 'negative'].mean()
#df_sentiment.to_excel("df_sentiment.xlsx") 


# Frequencies plots: VADER values 
sns.set(color_codes=True)

sns.distplot(df_sentiment['compound'], kde=False)
plt.title('Stock Trump Tweets', fontsize=18)
plt.xlabel('Compound Value', fontsize=16)
plt.ylabel('Frequencies', fontsize=16)
plt.savefig('compound_freq.png')

sns.distplot(df_sentiment['pos'], kde=False)
plt.title('Stock Trump Tweets', fontsize=18)
plt.xlabel('Positiv Valence', fontsize=16)
plt.ylabel('Frequencies', fontsize=16)
plt.savefig('positive_freq.png')

sns.distplot(df_sentiment['neg'], kde=False)
plt.title('Stock Trump Tweets', fontsize=18)
plt.xlabel('Negativ Valence', fontsize=16)
plt.ylabel('Frequencies', fontsize=16)
plt.savefig('negative_freq.png')


plt.figure(figsize=(30,19))
sns.lineplot(x = df_sentiment.index, y = 'compound', data = df_sentiment)
plt.title('Compound Values over time', fontsize=18)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Compound', fontsize=14)
plt.savefig('time_series_compound.png')
plt.show()

# Basic emotions: Describtive Stats
print("\nFear\n", df_sentiment.fear.describe())
print("\nAnger\n",df_sentiment.anger.describe())
print("\nAnticipation\n",df_sentiment.anticip.describe())
print("\nTrust\n",df_sentiment.trust.describe())
print("\nSurprise\n",df_sentiment.surprise.describe())
print("\nSadness\n",df_sentiment.sadness.describe())
print("\nDisgust\n",df_sentiment.disgust.describe())
print("\nJoy\n",df_sentiment.joy.describe())


plt.figure(figsize=(30,19))
sns.lineplot(x = df_sentiment.index, y = 'anger', data = df_sentiment)
plt.title('Anger over time', fontsize=18)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Anger', fontsize=14)
plt.savefig('time_series_anger.png')
plt.show()


####
# Prepare US-stock values and combine them with the dataframe

# source stock data:
# https://www.investing.com/indices/us-spx-500

stock = pd.read_csv("stock.csv", sep= ',', encoding='utf-8')
stock['Date'] = pd.to_datetime(stock['Date']).dt.date
stock.set_index('Date', inplace=True)
#stock['Change'] = stock['Change'].pipe(lambda x: (x - x.mean()) / x.std())   # Z-Standardizing winnings / losses 

# Concat stock and twitter data
df_calc = pd.concat([df_sentiment, stock['Change']], axis = 1)


##############
# Correlations between all variables

corr = df_calc.corr(method='pearson')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=.3, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5})
f.savefig('correlations.png')


corr[['Change']].sort_values(by=['Change'])

