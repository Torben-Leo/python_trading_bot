import re

import pandas as pd
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from pycaret.classification import *

# Todo Should sentimnet scores be weighted based on number of followers?
class TwitterClient:

    def __init__(self, consumer_key, consumer_secret, access_token_key, access_token_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token_key = access_token_key
        self.access_token_secret = access_token_secret
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token_key, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def get_tweets(self, query, count):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search_30_day('development', query=query, maxResults= count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                # saving the date
                parsed_tweet['date'] = tweet.created_at
                #saving the user_id
                parsed_tweet['author_id'] = tweet.author.id
                #saving the tweet id
                parsed_tweet['tweet_id'] = tweet.id
                #saving the user name
                parsed_tweet['user_name'] = tweet.user.name
                # saving the user name
                parsed_tweet['retweets'] = tweet.retweet_count
                # saving the user name
                parsed_tweet['number_followers'] = tweet.user.followers_count

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return pd.DataFrame(tweets)

        except tweepy.errors.TweepyException as e:
            # print error (if any)
            print("Error : " + str(e))


# creating object of TwitterClient Class
api = TwitterClient('AdjvBaBpABMWXxeuni1LHtWHR', 'TbfZfEP7KrCPpymrRTSfrz88cwcvHEA16jsiJ8EuVGEY8hKoBr',
                    '1326610440189784066-UwQoZIIvkOdhYdx74WvoZy8xh9htev',
                    'hQJ0iCIjUXeQ3t54iCFMailuDeyw8a1l5ey9BE0N4ccl3')
 # calling function to get tweets
tweets = api.get_tweets(query='Bitcoin -RT', count=100)
print(tweets)



