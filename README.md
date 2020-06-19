# Projekt-Aufgabe Big Data
## Sentiment Analyse

![alt text](https://github.com/skrause-data/trump_tweets/blob/master/wordcloud.png)



### Ziel des Projekts:
Sentiment Analysen: Trump Tweets auf ihre 

a) Valenz (Wertigkeit: positiv vs. negativ vs. neutral)
Korrelation bzw. Regression zwischen der Valenz von Trump Tweets und Börsen-Kursen beschreiben\

Compound (-1 bis 1) als unidimensionaler Wert aus den 3 Valenzwerten für die weiteren Analysen genutzt:\
„The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.“

b) auf Basis Emotionen untersuchen
Plutchik's wheel of emotions:

Furcht\
Wut\
Antizipation\
Vertrauen\
Überraschung\
Traurigkeit\
Ekel\
Freude

Eine weitere Ψ-Idee: Gerade Basis bzw. diskrete Emotionen (im Vergleich zu Valenzen) sind gute Prädiktoren für konkretes Verhalten (Spoiler: Diese Idee habe ich im Projekt nicht umgesetzt. Es wäre aber IMHO sehr sinnvoll sich theoriegeleitet diese Zusammenhänge mal genauer anzuschauen.)\

Beispiel Echo Chambers: 
•	Wut über politische Zustände führt zu mehr (Online)Debatten mit Personen, die sowohl ähnliche als auch unähnliche Meinungen, haben.
•	Angst führt dazu, dass man Informationen sucht, die der angstbesetze Meinung widersprechen

### Vorgehensweise:

1.	Download aller Tweets aus Archiv in eine csv und entsprechende Aufbereitung
1.1.	Quelle: http://www.trumptwitterarchive.com/archive
1.2.	CSV als Pandas DataFrame einlesen und Retweets, Links und Sonderzeichen entfernen

2.	Häufige Wörter zählen (exklusive stopwords) und in einer Wordcloud plotten.


3.	Sentiment Analyse

3.1.	Valenz der Tweets ermitteln mittels VADER und nltk
3.1.1.	 Häufigkeiten (Histogramme) plotten
3.1.2.	 Zeitreihe (mittels rolling mean) plotten

3.2.	Basis Emotionen mittels NRCLex (NRC Word-Emotion Association Lexicon)
3.2.1.	 Tokenizing und Stemming der Tweets
3.2.2.	 Analyse der stemmed Tweets mittels NRCLex
-	Funktion affect_frequencies gibt einen stetigen Wert zw. 0 und 1 für alle Emotionen für ein Wort aus.
-	Diese werden pro Tweet aufsummiert.
3.2.3.	 Zeitreihe (mittels rolling mean) plotten

4.	US-Börsenkurs (S&P 500 - Gewinne/Verluste)
-	Quelle: https://de.investing.com/indices/us-spx-500-historical-data
-	Gewinne/ Verluste z-standardisieren (Range -1 bis 1) um diese mit den Valenzen/Emotionen der Tweet gemeinsam zu plotten
    ->	to-do
