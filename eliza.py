import re
import pickle
import readrc
import tweepy
import normalizer as N


class Eliza():

    def __init__(self, model):
        self.username = 'ampeloss'
        with open(model, mode='rb') as f:
            self.lang = pickle.load(f)
        me = readrc.of(self.username)
        auth = tweepy.OAuthHandler(me.consumer_key, me.consumer_secret)
        auth.set_access_token(me.token, me.secret)
        self.api = tweepy.API(auth)

    def is_reply(self, status):
        text = status.text
        re_isreply = re.compile("@" + self.username)
        return re_isreply.match(text)

    def react(self, status):
        from_user = status.author.screen_name
        text = N.normalize(status.text)
        reply = self.lang.gen(text)
        print(from_user, text, reply)
        reply = "@{} {}".format(from_user, reply)[0:140]
        self.api.update_status(reply, status.id_str)


eliza = Eliza('./2Dcat.model')


class StreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if eliza.is_reply(status):
            eliza.react(status)


# entry
stream = tweepy.Stream(auth=eliza.api.auth, listener=StreamListener())
stream.userstream()
