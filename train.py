import sys
import signal
import pickle
import model
import argparse

parser = argparse.ArgumentParser(description='options')
parser.add_argument('-i', required=True, help='input text file')
parser.add_argument('-o', required=True, help='output model file (.pickle)')
parser.add_argument('--iteration', type=int, default=1000)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

sys.stderr.write("load LM\n")
lang = None
if args.resume:
    with open(args.o, mode='rb') as f:
        lang = pickle.load(f)
else:
    lang = model.Lang(args.i)

halt = False


def handler(num, frame):
    global halt
    halt = True

signal.signal(signal.SIGINT, handler)

sys.stderr.write("start learning\n")
M = 100
for _ in range(args.iteration//M):
    loss = lang.train(iterator=M)
    sys.stderr.write("loss={}\n".format(loss/M))
    with open(args.o, mode='wb') as f:
        pickle.dump(lang, f)
    if halt:
        break
