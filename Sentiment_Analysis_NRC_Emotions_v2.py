#!/usr/bin/env python
# coding: utf-8
# PIP-0815 Standard

# pip install nltk
# pip install pandas
# pip install wordcloud 
# pip install vaderSentiment

# pip install tweepy
# tweepy Dokumentation: http://docs.tweepy.org/en/latest/


'''
1. Bereite Tweets aus csv auf (meine Versuche mit der API waren wegen den Limits mir zu wenig)
'''


#################
#csv mit Datum für Tweets vom 15. Juni 2015 (Ankündigung Wahl) bis 28. Mai 2020
#################
#Quelle: http://www.trumptwitterarchive.com/archive

import pandas as pd
import html
df_raw = pd.read_csv("trump_2015.csv",sep='\t', encoding = "utf-8")



# entferne Retweets

df = df_raw[~df_raw["text"].str.contains("RT @", na=True)]
#df = df[~df["text"].str.contains(r'"@', na=True)] 

#Entferne Links und ersetz html Zeichen
df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True).replace(r'&amp;', '&', regex=True).replace(r'&gt;', '>', regex=True).replace(r'&lt;', '<', regex=True).replace(r'@realDonaldTrump', '', regex=True).replace(r'U.S.', 'UnitedStates', regex=True).replace(r'u.s.', 'UnitedStates', regex=True)



from collections import Counter
import string

import pandas as pd

from nltk.corpus import stopwords
from nltk import word_tokenize


stoplist = stopwords.words('english')

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

texts = df['text'].str.lower()
texts = texts.str.replace('[{}]'.format(string.punctuation), '')

pat = r'\b(?:{})\b'.format('|'.join(stoplist))
texts = texts.replace(pat, '', regex=True)


word_counts = Counter(word_tokenize('\n'.join(texts)))
print(texts)
word_counts.most_common(50)



# Gesamt-Deskriptiva
print(df.describe())



'''
2. Worthäufigkeit plotten
'''

import nltk

get_ipython().run_line_magic('matplotlib', 'inline')

# Worthäufigkeit
freqdist = nltk.FreqDist(df.text)

# Plotte 
freqdist.plot(25)



#Wordcloud
# siehe: https://amueller.github.io/word_cloud/index.html

import numpy as np
from PIL import Image

import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


temp = texts
stopw = " ".join(temp)
print(stopw)

stopwords = set(STOPWORDS)
mask = np.array(Image.open(r'background.png'))

wc = WordCloud(max_font_size=300, min_font_size=8, max_words=150, mask=mask, stopwords=stopwords, background_color='red', contour_width=2, contour_color='black').generate(stopw)


image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[15,15])
plt.imshow(wc.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")
plt.show()

wc.to_file(r'wordcloud.png')




'''
3. Sentiment Analysen
'''


#3.1. Valenz der Tweets ermitteln

#VADER (Valence Aware Dictionary and sEntiment Reasoner)
# scheinbar gut geeignet für Social Media Content und Slang
#siehe  https://github.com/cjhutto/vaderSentiment und
#       http://www.nltk.org/_modules/nltk/sentiment/vader.html

# Werte zw. -1 und +1 für positiv, negativ, neutral sowie ein compound Wert

'''
The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.
'''


'''
#Teste VADER

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

analyser = SentimentIntensityAnalyzer()

##Funktion

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<20} {}".format(sentence, str(score)))

sentiment_analyzer_scores("Trump is very dumb.")
sentiment_analyzer_scores("😀")
sentiment_analyzer_scores("💩")
'''

#txt = open('trumptweets.txt', 'r', encoding="utf8") 
#txt = open('trumptweets_seit_2015_bereinigt.txt', 'r', encoding="utf8") 
#lines = txt.readlines() 


#Füge die 4 Variablen aus VADER dem DataFrame hinzu

import numpy as np
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sentiment = df['text'].apply(lambda x: analyzer.polarity_scores(x))
df = pd.concat([df,sentiment.apply(pd.Series)],1)


# Gesamt-Deskriptiva
print(df.describe())
print(df.compound.describe())
print(df.pos.describe())
print(df.neg.describe())
print(df.neu.describe())




#Plotte VADER Compound Werte

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(color_codes=True)


#Histogramm für compound

sns.distplot(df['compound'], kde=False)
plt.title('Histogramm', fontsize=18)
plt.xlabel('Compound', fontsize=16)
plt.ylabel('Häufigkeit', fontsize=16)


#Histogramm für positiv

sns.distplot(df['pos'], kde=False)
plt.title('Histogramm', fontsize=18)
plt.xlabel('Positive Valenz', fontsize=16)
plt.ylabel('Häufigkeit', fontsize=16)


#Histogramm für negativ

sns.distplot(df['pos'], kde=False)
plt.title('Histogramm', fontsize=18)
plt.xlabel('Negative Valenz', fontsize=16)
plt.ylabel('Häufigkeit', fontsize=16)




#Plotte Compound über Zeit

#df.sort_values(by='created_at', inplace=True)
#df.index = pd.to_datetime(df['created_at'])

import numpy as np, pandas as pd; plt.close("all")
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


#rolling mean

df_plot = df.dropna(subset=['created_at'],inplace=True)
df_plot = df['created_at'] = pd.to_datetime(df['created_at'], infer_datetime_format=True)
df_plot= df.set_index('created_at', inplace=True)
df_plot= df.index.duplicated().sum()
df_plot= df = df[~df.index.duplicated()]
df_plot.sort_index(inplace=True)


df_plot['compound']
df_plot['mean'] = df['compound'].expanding().mean()
#df_plot['compound'].rolling("12h").mean()



df_plot.compound.rolling("96h").mean().plot(figsize=(100,80), linewidth=5, fontsize=60)
plt.ylim(-1, 1)
plt.xlabel('Jahr', fontsize=60)
plt.ylabel('Compound', fontsize=60)



# 2. Basis Emotionen bestimmen

# NRC Word-Emotion Association Lexicon 
# Erfasst Basis Emotionen, siehe auch
# https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

# Warum sind Basis Emotionen so wichtig?
# siehe: https://en.wikipedia.org/wiki/Robert_Plutchik und 
# https://en.wikipedia.org/wiki/Discrete_emotion_theory


'''
Plutchik's wheel of emotions:
    Angst
    Wut
    Antizipation
    Vertrauen
    Überraschung
    Traurigkeit
    Ekel
    Freude
'''



#2.1: Stemming jedes Tweets (Line)

# Wird empfohlen für die Basisemotionen um die Erkennung zu verbessern.
# Einen Lemmatizer nutze ich hier aus Zeitgründen nicht, wäre ggf. auch sinnvoll.


# pip install -U spacy
# python -m spacy download en

'''
# Spacy scheint sehr mächtig zu sein, funktioniert aber nicht hier. :/ Daher nutze ich hier dann doch PorterStemmer


import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

df['stem'] = df['text'].apply(lambda x: nlp(x))
print(df.stem)
'''

# Tokenizing und stemming der Tweets

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stop = stopwords.words('english')


df['token'] = df['text'].apply(lambda x :filter(None,x.split(" ")))
df['token'] = df['token'].apply(lambda x: [item for item in x if item not in stop])
df['stem']=df['token'].apply(lambda x : [ps.stem(y) for y in x])
df['stem'].apply(lambda word: word not in stop and word != '')
df['stemmed_sentence']=df['stem'].apply(lambda x : " ".join(x))




#2.2: Erkenne Basis Emotionen


# siehe  https://pypi.org/project/NRCLex/ und v.a.
#        http://saifmohammad.com/ 

# pip install NRCLex

from nrclex import NRCLex


'''
#Test
text_object = NRCLex('spider')
text_object.raw_emotion_scores

text_object = NRCLex('The zombie was eating brains.')
text_object.affect_frequencies

text_object = NRCLex('Donald is dumb')
text_object.affect_frequencies

text_object = NRCLex('Their hands have been atrociously mutilated.')
text_object.words
text_object.affect_frequencies

text_object = NRCLex('atrociously mutilate')
text_object.affect_frequencies


text_object = NRCLex('Menschen sterben schlimm.')
text_object.affect_frequencies

text_object = NRCLex('Mutter Beimer ist sehr dumm.')
text_object.words
# Deutsch scheint noch nicht implementiert zu sein.
'''


from nrclex import NRCLex

def emo(str):
    text_object = NRCLex(str)
    return text_object.affect_frequencies

'''
affect_frequencies gibt einen stetigen Wert zw. 0 und 1 für alle Emotionen für ein Wort aus.
'''


emotions = df['stemmed_sentence'].apply(lambda x: emo(x))
df = pd.concat([df,emotions.apply(pd.Series)],1)


# Gesamt-Deskriptiva
print("\nFurcht\n", df.fear.describe())
print("\nWut\n",df.anger.describe())
print("\nAntizipation\n",df.anticip.describe())
print("\nVertrauen\n",df.trust.describe())
print("\nÜberraschung\n",df.surprise.describe())
print("\nTraurigkeit\n",df.sadness.describe())
print("\nEkel\n",df.disgust.describe())
print("\nFreude\n",df.joy.describe())


#rolling means


df.fear.rolling("96h").mean().plot(figsize=(100,80), linewidth=5, fontsize=60)
plt.ylim(-1, 1)
plt.xlabel('Jahr', fontsize=60)
plt.ylabel('Angst', fontsize=60)


df.anger.rolling("96h").mean().plot(figsize=(100,80), linewidth=5, fontsize=60)
plt.ylim(-1, 1)
plt.xlabel('Jahr', fontsize=60)
plt.ylabel('Wut', fontsize=60)


df.disgust.rolling("96h").mean().plot(figsize=(100,80), linewidth=5, fontsize=60)
plt.ylim(-1, 1)
plt.xlabel('Jahr', fontsize=60)
plt.ylabel('Ekel', fontsize=60)


df.joy.rolling("96h").mean().plot(figsize=(100,80), linewidth=5, fontsize=60)
plt.ylim(-1, 1)
plt.xlabel('Jahr', fontsize=60)
plt.ylabel('Freude', fontsize=60)



'''
4. US-Börsenkurs (S&P 500 - Gewinne/Verluste) und Trump Valenz (Compound)
'''

# siehe Idee von https://towardsdatascience.com/covfefe-nlp-do-trumps-tweets-move-the-stock-market-42a83ab17fea


#import von Börsendaten
# Quelle: https://de.investing.com/indices/us-spx-500-historical-data

import pandas as pd
stock_raw = pd.read_csv("stock.csv", sep= ';', encoding='utf-8')

#error:  'unicodeescape' codec can't decode bytes in position 13410-13411: truncated \uXXXX escape




#Gewinne/Verluste z-standardizieren (-1 bis +1) um sie besser plotten zu können

temp = stock_raw.plusminus - stock_raw.plusminus.mean()
print(temp)
stock_raw.plusminus.std(ddof=0)

stock_raw = pd.concat([stock_raw, temp], axis=1, sort=False)

#To-do: z-Werte müssen noch in den DataFrame und dann mit Trump Compound Werten ploten


#Korrelation aus Trump Compound Werten und Gewinn/Verluste


import numpy as np
import pandas as pd

df.corr(method ='pearson') 

'''
To tos: 
    
- DateFrame Börsen-Kurs und Trump DataFrame zusammenführen um Korrelation berechnen zu können.
- Tweet-Compund Wert für einen Tag zusammenfassen um diese mit den Börsen Kursen zu matchen
- ggf. Tweets vorher nach Börsen-relevanten Wörtern filtern


'''

       
       
       
       
       
       
print("FERTIG!!!!")









