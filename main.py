# -*- coding: utf-8 -*-
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)
with open("TIME MACHINE.txt") as f:
    time_machine = f.read()

#print(time_machine[0:500])
#print(time_machine[-38075:-37500])
#time_machine = time_machine[:-38083]
character_list  = list(set(time_machine))
vocab_size= len(character_list)
#print (character_list)
#print("Lengrh of vocab: %s" % vocab_size)

character_dict = {}
for e,char in enumerate(character_list):
    character_dict[char] = e

#print(character_dict)
time_numerical = [character_dict[char] for char in time_machine]
#print(len(time_numerical))
#print(time_numerical[:20])
#print("".join([character_list[idx] for idx in time_numerical[:39]]))
def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list),vocab_size),ctx = ctx)
    for i,idx in enumerate(numerical_list):
        result[i,idx] = 1.0
    return result
print (one_hots(time_numerical[:2]))
def textify(embedding):
    result=""
    indices = nd.argmax(embdding,axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result
textify(one_hots(time_numerical[0:40]))
#data tranning
seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(time_numerical) - 1) // seq_length
dataset = one_hots(time_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
textify(dataset[0])
#
batch_size =32 and
print('# of sequences in dataset: ', len(dataset))
num_batches = len(dataset) // batch_size
print('# of batches: ', num_batches)
train_data = dataset[:num_batches*batch_size].reshape((batch_size, num_batches, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 0, 1)
train_data = nd.swapaxes(train_data, 1, 2)
print('Shape of data set: ', train_data.shape)
for i in range(3):
    print("***Batch %s:***\n %s \n %s \n\n" % (i, textify(train_data[i, :, 0]), textify(train_data[i, :, 1])))
labels = one_hots(time_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((batch_size, num_batches, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 0, 1)
train_label = nd.swapaxes(train_label, 1, 2)
print(train_label.shape)
print(textify(train_data[10, :, 3]))
print(textify(train_label[10, :, 3]))
