from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import reddit_api_connection
import pandas as pd
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
#nltk.download('vader_lexicon')
sns.set(style='darkgrid', context='talk', palette='Dark2')


class SentimentAnalyser:

    def __init__(self, reddit_client):
        # initialize SentimentAnalyser Object
        self.sia = SIA()
        # list to store result of sentiment analysis on headlines
        self.results = pd.DataFrame
        # initilaize the reddit clinet of the class
        self.reddit_client = reddit_client

    def get_polscore(self):
        temp = []
        # calculate the polscore for each headline downloaded from the API
        for line in self.reddit_client.headlines:
            pol_score = self.sia.polarity_scores(line[0])
            pol_score['headline'] = line[0]
            pol_score['date'] = line[1]
            temp.append(pol_score)
        # the dataframe consists of four different six different columns compound, headline, neg, neu, pos, label
        df = pd.DataFrame(temp)
        # create label column, based on compounded score, all values with a sentiment score smaller -.3 are considered
        # negative while values larger .3 are considered positive
        df['label'] = 0
        df.loc[df['compound'] > .2, 'label'] = 1
        df.loc[df['compound'] < -.2, 'label'] = -1
        df.head()
        self.results = df

    def show_positive_headlines(self):
        # absolute counts
        print(self.results.label.value_counts())
        # relative counts
        print(self.results.label.value_counts(normalize=True) * 100)
        # print bar plot
        fig, ax = plt.subplots(figsize=(8, 8))
        counts = self.results.label.value_counts(normalize=True) * 100
        sns.barplot(x=counts.index, y=counts, ax=ax)
        ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
        ax.set_ylabel("Percentage")
        plt.show()


reddit = reddit_api_connection.RedditClient(client_id='6eThO9Cr1LNuhduKYWH-Fw',
                                            client_secret='V4XQ33LOktsRDJ9dsg-ZL6AgJVk7aQ',
                                            user_agent='tom_forest')
reddit.get_headlines('bitcoin')
sa = SentimentAnalyser(reddit_client= reddit)
sa.get_polscore()
sa.show_positive_headlines()
