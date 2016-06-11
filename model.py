import numpy as np
import random
from chainer import Variable, optimizers
from chainer import Chain
import chainer.functions as F
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


def align(us, vs):
    """
    us = [[1,2,2,0], [1,2,0]]
    vs = [[3,3,3,3,0], [2,0]]
    xs = [[1,2,2,0,3,3,3,3], [1,2,0,2,0,0,0,0]]
    ys = [[-,-,-,3,3,3,3,0], [-,-,2,0,-,-,-,-]]
    """
    n = len(us)
    m = max(len(u) + len(v) - 1 for u, v in zip(us, vs))
    xs = np.zeros((n, m), dtype=np.int32)
    ys = np.zeros((n, m), dtype=np.int32)
    for x, y, u, v in zip(xs, ys, us, vs):
        k = len(u)
        t = len(v)
        x[0:k] = u
        x[k:k+t-1] = v[:-1]
        y[:] = -1
        y[k-1:k+t-1] = v
    return xs, ys


class Lang(Chain):

    def __init__(self, trainfile):
        self.alphabet = {}  # char => id
        self.rev_alphabet = {}  # id => char
        self.train_data = []

        self.load(trainfile)
        self.model = network.LMNet(len(self.alphabet))

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
        eos = self.alphabet["<eos>"]

        lines = []
        with open(trainfile, mode='r') as f:
            for line in f:
                lines.append(line.strip())

        for i in range(len(lines)//2):
            a = self.sentence_to_vector(lines[i * 2], addition=True)
            b = self.sentence_to_vector(lines[i * 2 + 1], addition=True)
            a = a + [eos]
            b = b + [eos]
            self.train_data.append({"input": a, "output": b})

    def read(self, id):
        x = Variable(np.array([id], dtype=np.int32).reshape((1, 1)))
        return choose(list(self.model(x).data[0]))

    def gen(self, sentence):
        eos = self.alphabet["<eos>"]
        u = self.sentence_to_vector(sentence) + [eos]
        self.model.reset_state()
        for a in u:
            y = self.read(a)
        ret = [y]
        while ret[-1] != eos:
            ret.append(self.read(ret[-1]))
            if len(ret) > 130:
                break
        ret = filter(lambda id: id != eos, ret)
        ret = map(lambda id: self.rev_alphabet[id], ret)
        ret = ''.join(ret)
        return ret

    def error(self, us, vs):
        xs, ys = align(us, vs)
        xs = np.transpose(xs)
        ys = np.transpose(ys)
        loss = 0.0
        self.model.reset_state()
        for x, y in zip(xs, ys):
            x = Variable(x)
            y = Variable(y)
            loss += F.softmax_cross_entropy(self.model(x, train=True), y)
        return loss

    def train(self, batch=100, cx=0):
        n = len(self.train_data)
        order = np.random.permutation(n)

        indicies = [order[i % n] for i in range(cx * batch, (cx + 1) * batch)]
        us = [self.train_data[i]["input"] for i in indicies]
        vs = [self.train_data[i]["output"] for i in indicies]
        loss = self.error(us, vs)

        self.model.zerograds()
        loss.backward()
        loss.unchain_backward()
        self.opt.update()

        return loss.data / batch
