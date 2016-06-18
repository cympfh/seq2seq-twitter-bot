import time
import pickle
import model
import argparse

parser = argparse.ArgumentParser(description='options')
parser.add_argument('-i', required=True, help='input text file')
parser.add_argument('-o', required=True, help='output model file (.pickle)')
parser.add_argument('--iteration', type=int, default=1000)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

lang = None
if args.resume:
    print("load model: {}".format(args.o))
    with open(args.o, mode='rb') as f:
        lang = pickle.load(f)
else:
    lang = model.Lang(args.i)

print("start learning")
for cx in range(args.iteration):
    start = time.time()
    loss = lang.train(batch=100)
    elapsed_time = time.time() - start
    print("#{} loss={} ({} sec)".format(cx, loss, elapsed_time))
    if cx % 10 == 9:
        print("saving this model as {}".format(args.o))
        with open(args.o, mode='wb') as f:
            pickle.dump(lang, f)
            print("done")
    time.sleep(10)
