
# env

- python==3.5.1
- chainer==1.8.2

# files

- `/eliza`: a network and a language model
- `/script`: scripts to acquire corpus from Twitter

# model

- model = `x -> LSTM -> Linear -> x'` where `x` and `x'` are chars
- let input = `w1 w2 .. wn`
- she reads `<a> wn .. w2 w1 <b> w1 w2 .. wn <c>` in order
- after read `<c>`, she outputs chars
    - `v1 v2 .. vm <d>`
    - her reply is `v1 v2 .. vm`


