from chainer import Chain
import chainer.functions as F
import chainer.links as L


class LMNet(Chain):

    def __init__(self, alphabetsize):
        super(LMNet, self).__init__(
            embed=L.EmbedID(alphabetsize, 1000),
            lstm=L.LSTM(1000, 200),
            lin1=L.Linear(200, 100),
            lin2=L.Linear(100, alphabetsize),
        )

    def __call__(self, x, train=False):
        x = self.embed(x)
        h = F.dropout(self.lstm(x), train=train, ratio=0.1)
        h = F.tanh(self.lin1(h))
        y = self.lin2(h)
        return y

    def reset_state(self):
        self.lstm.reset_state()
