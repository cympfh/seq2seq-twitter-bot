import sys
import numpy as np
import random
import chainer
from chainer import serializers, cuda, Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import network

special_chars = ["<a>", "<b>", "<c>", "<d>", "<♡>"]

class Lang(Chain):

    def __init__(self, trainfile):
        self.alphabet = {} # char => id
        self.rev_alphabet = {} # id => char
        self.train_data = []

        self.load(trainfile)
        self.model = L.Classifier(network.LMNet(len(self.alphabet)))

        self.opt = optimizers.SGD()
        self.opt.setup(self.model)

    def add_alphabet(self, a):
        if a not in self.alphabet:
            m = len(self.alphabet)
            self.alphabet[a] = m
            self.rev_alphabet[m] = a
        return self.alphabet[a]

    def sentence_to_vector(self, s, addition=False):
        if addition:
            return [ self.add_alphabet(a) for a in list(s) ]
        else:
            unk = self.alphabet["<♡>"]
            return [ self.alphabet[a] if a in self.alphabet else unk for a in list(s) ]

    def load(self, trainfile):
        for a in special_chars:
            self.add_alphabet(a)

        lines = []
        with open(trainfile, mode='r') as f:
            for line in f:
                lines.append(line.strip())
        for i in range(len(lines)//2):
            a = self.sentence_to_vector(lines[i * 2], addition=True)
            b = self.sentence_to_vector(lines[i * 2 + 1], addition=True)
            self.train_data.append({ "input": a, "output": b })

    def read(self, id, id2=None):
        x = Variable(np.array([ id ], dtype=np.int32).reshape((1,1)))
        if id2 is None:
            return np.argmax(self.model.predictor(x).data)
        else:
            t = Variable(np.array([ id2 ], dtype=np.int32).reshape((1,)))
            return self.model(x, t)

    def gen(self, in_str):
        ret = []
        a = self.alphabet["<a>"]
        b = self.alphabet["<b>"]
        c = self.alphabet["<c>"]
        d = self.alphabet["<d>"]

        u = self.sentence_to_vector(in_str)
        n = len(u)

        self.model.predictor.reset_state()
        self.read( a )
        for i in range(n): self.read(u[n-i-1])
        self.read( b )
        for i in range(n): self.read(u[i])

        ret.append( self.read( c ))
        while ret[-1] != d:
            ret.append( self.read( ret[-1]))
            if len(ret) > 100:
                break
        ret = filter(lambda id: id!=a and id!=b and id!=c and id!=d, ret)
        ret = map(lambda id: self.rev_alphabet[id], ret)
        ret = ''.join(ret)
        return ret

    def train(self, iterator=1000):
        a = self.alphabet["<a>"]
        b = self.alphabet["<b>"]
        c = self.alphabet["<c>"]
        d = self.alphabet["<d>"]

        for cx in range(iterator):
            sys.stderr.write("\r{}/{}".format(cx + 1, iterator))
            idx = random.randint(0, len(self.train_data) - 1)
            u = self.train_data[idx]["input"]
            v = self.train_data[idx]["output"]
            n = len(u)
            m = len(v)

            loss = 0
            self.model.predictor.reset_state()
            self.read( a )
            for i in range(n): self.read(u[n-i-1])
            self.read( b )
            for i in range(n): self.read(u[i])
            last = c

            for i in range(m):
                loss += self.read(last, v[i])
                last = v[i]
            loss += self.read(last, d)

            self.model.zerograds()
            loss.backward()
            loss.unchain_backward()
            self.opt.update()

        sys.stderr.write("\n")
