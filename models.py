
from keras import models, layers


inp = layers.Input((None, 256))
x = layers.LSTM(512, return_sequences=True)(inp)
x = layers.LSTM(256)(x)
x = layers.Dense(32, name='rnn2_output')(x)
rnn2 = models.Model(inp, x)
rnn2.trainable = False
rnn2_output = rnn2.output

inp2 = layers.Input((None, 32))
x = layers.LSTM(256, return_sequences=True)(inp2)
x = rnn2(x)
x = layers.Dense(1)(x)
rnn1 = models.Model(inp2, x)
rnn1.summary()
