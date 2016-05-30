from chainer import Chain
import chainer.functions as F
import chainer.links as L


class LMNet(Chain):

    def __init__(self, alphabetsize):
        super(LMNet, self).__init__(
            embed=L.EmbedID(alphabetsize, 1000),
            lstm=L.LSTM(1000, 100),
            lin1=L.Linear(100, 50),
            lin2=L.Linear(50, alphabetsize),
        )

    def __call__(self, x):
        x = self.embed(x)
        h1 = F.relu(self.lstm(x))
        h2 = F.sigmoid(self.lin1(h1))
        y = self.lin2(h2)
        return y

    def reset_state(self):
        self.lstm.reset_state()
