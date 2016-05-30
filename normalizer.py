import re

re_at = re.compile("@[a-zA-Z0-9_]*")
def normalize(sentence):
    sentence = re.sub(re_at, "", sentence)
    sentence = sentence.strip()
    return sentence
