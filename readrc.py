import os
import re


class TwitterToken():
    def __init__(self, ck, cs, to, se):
        self.consumer_key = ck
        self.consumer_secret = cs
        self.token = to
        self.secret = se


def of(username):
    user = {
        "consumer_key": None,
        "consumer_secret": None,
        "token": None,
        "secret": None
    }
    re_ck = re.compile(r"^consumer_key:")
    re_cs = re.compile(r"^consumer_secret:")
    re_to = re.compile(r"^token:")
    re_se = re.compile(r"^secret:")
    with open(os.environ['HOME'] + '/.twurlrc') as f:
        sw = False
        for line in f:
            line = line.strip()
            if line == username + ':':
                sw = True
            elif sw and user["consumer_key"] is None and re_ck.match(line):
                user["consumer_key"] = line.split(' ')[1]
            elif sw and user["consumer_secret"] is None and re_cs.match(line):
                user["consumer_secret"] = line.split(' ')[1]
            elif sw and user["token"] is None and re_to.match(line):
                user["token"] = line.split(' ')[1]
            elif sw and user["secret"] is None and re_se.match(line):
                user["secret"] = line.split(' ')[1]
    t = TwitterToken(
            user["consumer_key"],
            user["consumer_secret"],
            user["token"],
            user["secret"])
    return t
