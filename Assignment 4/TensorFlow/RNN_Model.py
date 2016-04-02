import tensorflow as tf
import numpy as np
from os import path, makedirs
from math import ceil

def round_up(val, base):
    return int(ceil(float(val)/base)*base)

class RNN_Model:

    def __init__(self, vocab_size, hidden_size, seq_length=25, temperature=1):
        self.inputs = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="inputs")
        self.targets = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="targets")
        self.init_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        initializer = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope("RNN") as scope:
            hs_t = self.init_state
            ys = []
            for t, xs_t in enumerate(tf.split(0, self.seq_length, self.inputs)):
                if t > 0: scope.reuse_variables()  # Reuse variables
                Wxh = tf.get_variable("Wxh", [vocab_size, hidden_size], initializer=initializer)
                Whh = tf.get_variable("Whh", [hidden_size, hidden_size], initializer=initializer)
                Why = tf.get_variable("Why", [hidden_size, vocab_size], initializer=initializer)
                bh = tf.get_variable("bh", [hidden_size], initializer=initializer)
                by = tf.get_variable("by", [vocab_size], initializer=initializer)

                hs_t = tf.tanh(tf.matmul(xs_t, Wxh) + tf.matmul(hs_t, Whh) + bh)
                ys_t = tf.matmul(hs_t, Why) + by
                ys.append(ys_t)

        self.hprev_val = np.zeros([1, hidden_size])
        self.hprev = hs_t

        self.output_softmax = tf.nn.softmax(ys[-1] / temperature)  # Get softmax for sampling
        self.outputs = tf.concat(0, ys)

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.targets))

        self.saver = tf.train.Saver()

        self.create_gradients()

    def create_gradients(self):
        # Minimizer
        minimizer = tf.train.AdamOptimizer()
        grads_and_vars = minimizer.compute_gradients(self.loss)

        # Gradient clipping
        grad_clipping = tf.constant(5.0, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))

        # Gradient updates
        self.updates = minimizer.apply_gradients(clipped_grads_and_vars)

        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        if path.exists("saved/"):
            if path.exists("saved/trained_rnn.ckpt"):
                print "Reloading model from disk"
                self.saver.restore(self.sess, "saved/trained_rnn.ckpt")
        else:
            makedirs("saved/")

    def run_rnn(self, inputs):
        softmax_output, self.hprev_val = \
            self.sess.run([self.output_softmax, self.hprev],
                          feed_dict={self.inputs: inputs,
                                     self.init_state: self.hprev_val})

        return softmax_output, self.hprev_val

    def update_rnn(self, inputs, targets):
        self.hprev_val, loss_val, _ = \
            self.sess.run([self.hprev, self.loss, self.updates],
                          feed_dict={self.inputs: inputs,
                                     self.targets: targets,
                                     self.init_state: self.hprev_val})
        return loss_val

    def reset_state(self):
        self.hprev_val = np.zeros_like(self.hprev_val)

    def save(self):
        self.saver.save(self.sess, "saved/trained_rnn.ckpt")

    def generate_sample(self, inputs, sample_length):
        self.reset_state()
        ixes = []
        if len(inputs) != self.seq_length:
            inputs = self.pad_input(inputs)
        linputs = np.split(np.asarray(inputs), len(inputs)/self.seq_length)
        for inputs in linputs:
            inputs = list(inputs)
            for t in range(sample_length):
                input_encoded = self.one_hot(inputs)
                softmax_outputs = self.run_rnn(input_encoded)[0]
                ix = np.random.choice(range(self.vocab_size), p=softmax_outputs.ravel())
                ixes.append(ix)
                inputs = inputs[1:] + [ix]
        return ixes


    def pad_input(self, input):
        inputlen = len(input)
        padv = round_up(inputlen, self.seq_length) - inputlen
        res = list(np.hstack((np.ones(padv)*(self.vocab_size), input)).astype('int32'))
        return res


    def one_hot(self, v):
        dic = np.vstack((np.eye(self.vocab_size), np.zeros(self.vocab_size)))
        return dic[v]