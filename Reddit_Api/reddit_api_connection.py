from IPython import display
import praw
from datetime import datetime


# Additional classes for streaming and sentiment analysis not finished -> for future improvements

class RedditClient:
    # TODO does set automatically prevent duplicates in the headline set?
    def __init__(self, client_id, client_secret, user_agent):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.connection = praw.Reddit(client_id=self.client_id,
                                      client_secret=self.client_secret,
                                      user_agent=self.user_agent)
        self.headlines = set()

    def get_headlines(self, subreddit):
        for submission in self.connection.subreddit(subreddit).new(limit=None):
            self.headlines.add((submission.title, datetime.fromtimestamp(submission.created_utc)))
            display.clear_output()
