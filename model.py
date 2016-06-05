import numpy as np
import random
from chainer import Variable, optimizers
from chainer import Chain
import chainer.links as L
import network

special_chars = ["<eos>", "<unk>"]


def choose(ls):
    a = list(zip(ls, range(len(ls))))
    a.sort()
    a.reverse()
    m = 1 if random.random() < 0.8 else 2
    a = list(map(lambda p: p[1], a[0:m]))
    i = random.randint(0, m-1)
    return a[i]


class Lang(Chain):

    def __init__(self, trainfile):
        self.alphabet = {}  # char => id
        self.rev_alphabet = {}  # id => char
        self.train_data = []

        self.load(trainfile)
        self.model = L.Classifier(network.LMNet(len(self.alphabet)))

        self.opt = optimizers.AdaGrad(lr=0.004)
        self.opt.setup(self.model)

    def add_alphabet(self, a):
        if a not in self.alphabet:
            m = len(self.alphabet)
            self.alphabet[a] = m
            self.rev_alphabet[m] = a
        return self.alphabet[a]

    def sentence_to_vector(self, s, addition=False):
        if addition:
            return [self.add_alphabet(a) for a in list(s)]
        else:
            unk = self.alphabet["<unk>"]
            return [self.alphabet[a] if a in self.alphabet else unk
                    for a in list(s)]

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
            self.train_data.append({"input": a, "output": b})

    def read(self, id, id2=None):
        x = Variable(np.array([id], dtype=np.int32).reshape((1, 1)))
        if id2 is None:
            return choose(list(self.model.predictor(x).data[0]))
        else:
            t = Variable(np.array([id2], dtype=np.int32).reshape((1,)))
            return self.model(x, t)

    def gen(self, in_str):
        ret = []
        eos = self.alphabet["<eos>"]

        u = self.sentence_to_vector(in_str)
        n = len(u)

        self.model.predictor.reset_state()
        for i in range(n):
            self.read(u[n-i-1])
        # for i in range(n):
        #     self.read(u[i])
        ret.append(self.read(eos))
        while ret[-1] != eos:
            ret.append(self.read(ret[-1]))
            if len(ret) > 130:
                break
        ret = filter(lambda id: id != eos, ret)
        ret = map(lambda id: self.rev_alphabet[id], ret)
        ret = ''.join(ret)
        return ret

    def train(self, iterator=1000):
        eos = self.alphabet["<eos>"]
        sum_loss = 0.0

        for cx in range(iterator):
            idx = random.randint(0, len(self.train_data) - 1)
            u = self.train_data[idx]["input"]
            v = self.train_data[idx]["output"]
            n = len(u)
            m = len(v)

            self.model.predictor.reset_state()
            for i in range(n):
                self.read(u[n-i-1])
            # for i in range(n):
            #     self.read(u[i])

            loss = 0
            last = eos
            for i in range(m):
                loss += self.read(last, v[i])
                last = v[i]
            loss += self.read(last, eos)

            self.model.zerograds()
            loss.backward()
            loss.unchain_backward()
            self.opt.update()

            sum_loss += loss.data
        return sum_loss
