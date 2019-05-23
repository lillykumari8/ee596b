import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class Model():
    def __init__(self, batch_size, seq_len, vocab_size, rnn_size, rnn_type, num_layers, use_dropout, lr, is_training, input_keep_prob, outptut_keep_prob, use_embedding, embedding_size, max_grad_clip):

        if is_training:
            self.batch_size = batch_size
            self.seq_len = seq_len
        else:
            self.batch_size = 1
            self.seq_len = 1
        #self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.lr = tf.constant(lr)
        self.is_training = is_training
        # self.is_testing = is_testing
        self.input_keep_prob = input_keep_prob
        self.outptut_keep_prob = outptut_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        self.max_grad_clip = max_grad_clip

        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='X')
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='Y')


        def get_cell(rnn_size, input_keep_prob, outptut_keep_prob, is_training, use_dropout):
            cell = rnn.BasicRNNCell(rnn_size)
            if is_training:
                if use_dropout:
                    cell = rnn.DropoutWrapper(cell, input_keep_prob=self.input_keep_prob, outptut_keep_prob=self.outptut_keep_prob)
            return cell


        
        self.cell = cell = rnn.MultiRNNCell([get_cell(self.rnn_size, self.input_keep_prob, self.outptut_keep_prob, self.is_training, self.use_dropout)
                        for _ in range(self.num_layers)], state_is_tuple=True)
#         cell = self.cell

        print ("HELLOI")
        self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        if self.use_embedding:
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size])
            rnn_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        else:
            rnn_inputs = tf.one_hot(self.inputs, self.vocab_size)


        if is_training and use_dropout:
            rnn_inputs = tf.nn.dropout(rnn_inputs, self.input_keep_prob)

        rnn_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(value=rnn_inputs, num_or_size_splits=self.seq_len, axis=1)]

        outputs, final_state = rnn.static_rnn(cell, rnn_inputs, initial_state=self.initial_state, scope='model')

        flatten_outputs = tf.reshape(tf.concat(values=outputs, axis=1), [-1, self.rnn_size])
        self.final_state = final_state
        flatten_targets = tf.reshape(tf.concat(values=self.targets, axis=1), [-1])

        weights = tf.get_variable("weights_last_layer", [self.rnn_size, self.vocab_size])
        baises = tf.get_variable("bias_last_layer", [self.vocab_size])

        self.logits = tf.matmul(flatten_outputs, weights) + baises
        self.probabilities = tf.nn.softmax(self.logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=flatten_targets)
        self.final_loss = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.final_loss, tvars), self.max_grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))




    def sample(self, sess, chars, vocab, n, starting_string):

        state = sess.run(self.cell.zero_state(1, dtype=tf.float32))
        output_seq = starting_string

        for char in starting_string[:-1]:
            x = np.zeros((1,1))
            x[0,0] = vocab[char]
            state = sess.run(self.final_state, {self.inputs: x, self.initial_state: state})


        last_char = starting_string[-1]
        for i in range(n):
            x = np.zeros((1,1))
            x[0,0] = vocab[last_char]
            [state, probabilities] = sess.run([self.final_state, self.probabilities], {self.inputs: x, self.initial_state: state})
            prob = probabilities[0]
            pred_id = np.argmax(prob)
            print (type(pred_id))
            pred_char = chars[int(pred_id)]
            output_seq += str(pred_char)
            last_char = pred_char
        return output_seq

