import chainer
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

class LMNet(Chain):
    def __init__(self, alphabetsize):
        super(LMNet, self).__init__(
            embed = L.EmbedID(alphabetsize, 100),
            l1 = L.LSTM(100, 30),
            l2 = L.Linear(30, alphabetsize),
        )
    def __call__(self, x):
        x = self.embed(x)
        h = self.l1(x)
        y = self.l2(h)
        return y

    def reset_state(self):
        self.l1.reset_state()

