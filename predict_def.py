import string
import numpy as np
import pandas as pd
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D

def create_vocab_set():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'] + [' '] )
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check

def model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    # conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
    #                       border_mode='valid', activation='relu')(conv2)

    # conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
    #                       border_mode='valid', activation='relu')(conv3)

    # conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],---------
    #                       border_mode='valid', activation='relu')(conv4)
    # conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv2)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model
def model2(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):                                                  # For Character Embedding use this model instead of above model
    d = 300             #Embedding Size
    Embedding_layer  = Embedding(vocab_size+1, d, input_length=maxlen)
    inputs = Input(shape=(maxlen,), name='input', dtype='float32')
    embed = Embedding_layer(inputs)
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                             border_mode='valid', activation='relu',
                             input_shape=(maxlen, d))(embed)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    # conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
    #                       border_mode='valid', activation='relu')(conv2)

    # conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
    #                       border_mode='valid', activation='relu')(conv3)

    # conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],---------
    #                       border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv2)
    conv = Flatten()(conv)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv))
    # z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model
def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)
        #input_data = encode_data2(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)
        yield (input_data, y_sample)

def encode_data(x, maxlen, vocab, vocab_size, check):

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower())
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data
def encode_data2(x, maxlen, vocab, vocab_size, check):              # For character embedding use this function instead of encode_data

    input_data = np.zeros((len(x), maxlen))
    for dix, sent in enumerate(x):
        counter = 0
        chars = list(sent.lower().replace(" ", ""))
        chars2 = []
        for i in range(len(chars)-1):
            chars2.append(chars[i] + chars[i+1])
        for i,c in enumerate(chars2):
            if counter >= maxlen:
                pass
            else:
                if c in check:
                    counter += 1
                    ix = vocab[c]
                    input_data[dix,counter-1] = ix

    return input_data
def shuffle_matrix(x, y):
    print (x.shape, y.shape)
    stacked = np.hstack((np.matrix(x), y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi
