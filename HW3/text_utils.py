import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_len):
        # pass
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        txt_file = os.path.join(data_dir, "shakespeare.txt")
        output_file = os.path.join(data_dir, "output_shakespeare.npy")
        vocab_file = os.path.join(data_dir, "vocab_shakespeare.pkl")
        
        self.process_data(txt_file, vocab_file, output_file)
        self.load_preprocessed_data(vocab_file, output_file)
            
        self.generate_batches()

    def process_data(self, txt_file, vocab_file, output_file):

        with codecs.open(txt_file, "r", encoding='utf-8') as f1:
            txt_data = f1.read()

        self.char = list(set(txt_data))
        self.vocab_size = len(self.char)

        self.vocab = {}
        for i, ch in enumerate(self.char):
            self.vocab[ch] = i

        with open(vocab_file, 'wb') as f2:
            cPickle.dump(self.vocab, f2)

        self.output = list(map(self.vocab.get, txt_data))
        self.output = np.asarray(self.output)

        np.save(output_file, self.output)


    def load_preprocessed_data(self, vocab_file, output_file):
        with open(vocab_file, 'rb') as f1:
            self.vocab = cPickle.load(f1)

        self.vocab_size = len(self.vocab.keys())

        self.output = np.load(output_file)


    def generate_batches(self):

        self.num_batches = self.output.size // (self.seq_len * self.batch_size)
        self.output = self.output[:self.num_batches * self.seq_len * self.batch_size]

        self.output = self.output.reshape((self.batch_size, -1))

        np.random.shuffle(self.output)
        for i in range(0, self.output.shape[1], self.seq_len):
            x = self.output[:, i : i + self.seq_len]
            y = np.zeros_like(x)

            y[:, :-1] = x[:, 1:]
            y[:, -1] = x[:, 0]

            yield x, y


