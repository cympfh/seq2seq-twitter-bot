
# env

- python==3.5.1
- chainer==1.8.2

# files

- `/eliza`: model and networks using chainer
- `/script`: script to acquire scripts from Twitter

# model

- `x -> LSTM -> Linear -> x'`
- let input = `w1 w2 .. wn`
- model reads `<a>, wn, .. w2, w1, <b> w1, w2, .. wn <c>`
- after read `<c>`, she outputs
    - `v1 v2 .. vm <d>`
    - let output = `v1 v2 .. vm`


