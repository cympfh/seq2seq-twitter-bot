import sys
import pickle
# import model

with open(sys.argv[1], mode='rb') as f:
    lang = pickle.load(f)
    while True:
        sys.stdout.write("> ")
        s = lang.gen(input())
        print(s)
