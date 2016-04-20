import numpy as np
import re
from os import path, rename
from utils import *


class RNN_Model:
    def __init__(self, hidden_size, chars, temperature=1, max_updates=5000):
        '''
            Creates a single-layer RNN with the parameters provided
        :param hidden_size: Size of the hidden state
        :param chars: RNN vocabulary
        :param temperature: Softmax temperature, defaults to 1
        :param max_updates: Maximum number of updates between saves to disk, defaults to 5000
        '''
        if path.exists("saved_rnn.npz"):
            if raw_input("Found saved model. Load? [Y/n] ").strip().lower() != 'n':
                with np.load(open("saved_rnn.npz")) as a:
                    for var in a.files:
                        exec("self.%s = a['%s']" %(var, var)) # ~ I too like to live dangerously ~
                print_warning("[I] Model loaded from disk.")
                self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
                self.ix_to_char = {i: ch for i, ch in enumerate(chars)}
                return
            else:
                rename("saved_rnn.npz", "saved_rnn.npz.bkp")

        vocab_size = len(chars)
        # Model Hyper-parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temperature = temperature

        # Model Auxiliary Variables
        self.update_count = 0
        self.max_updates = max_updates
        self.saves = 0

        # Encoding/Decoding Dictionaries
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # Model Parameters
        self.hprev = np.zeros((hidden_size,1))
        self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.by = np.zeros((vocab_size, 1)) # output bias

        # Memory Variables for Gradient
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        print_warning("[I] Model created.")




    def update_rnn(self, inputs, targets, learning_rate=1e-1):
        '''
            Performs a parameter update using the inputs and targets provided
        :param inputs: Training inputs
        :param targets: Training targets
        :param learning_rate: Learning rate for Adagrad
        :return: Training loss
        '''
        inputs = [self.char_to_ix[ch] for ch in inputs]
        targets = [self.char_to_ix[ch] for ch in targets]

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        loss = 0
        # Forward Pass
        for t in xrange(len(inputs)):
            xs[t] = self.one_hot(inputs[t])
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]/self.temperature) / np.sum(np.exp(ys[t]/self.temperature))
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

        # Backward Propagation
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(xrange(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        # Clip gradient
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update parameters with Adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):

            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        # Save rnn to disk every few updates
        self.update_count += 1
        if self.update_count % self.max_updates == 0:
            self.save_rnn()
            self.update_count = 0

        return loss


    def sample_rnn(self, seed, sample_length, temperature=1):
        '''
            Samples a text from the RNN using seed as the initial exciter.
        :param seed: Initial exciter
        :param sample_length: Length of the sample to be produced
        :param temperature: Softmax temperature
        :return: Produced sample
        '''
        self.temperature = temperature
        x = self.one_hot(self.char_to_ix[seed])
        sample = ''
        for t in xrange(sample_length):
            self.hprev = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.hprev) + self.bh)
            y = np.dot(self.Why, self.hprev) + self.by
            p = np.exp(y/self.temperature) / np.sum(np.exp(y/self.temperature))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            sample += self.ix_to_char[ix]
            x = self.one_hot(ix)

        return sample


    def excite_rnn(self, chr):
        ix = self.one_hot(chr)
        self.hprev = np.tanh(np.dot(self.Wxh, ix) + np.dot(self.Whh, self.hprev) + self.bh)


    def get_prob(self, chr):
        self.excite_rnn(chr)
        y = np.dot(self.Why, self.hprev) + self.by
        den = np.sum(np.exp(y/self.temperature))
        p = np.exp(y/self.temperature) / den
        return p, den

    def complete_phrase(self, phrase, temperature=1, minlenght=100):
        '''
            Completes a given phrase using RNN samples.
        :param phrase: Phrase to be completed
        :return: Phrase with completion
        '''
        self.temperature = temperature
        phrase_ix = [self.char_to_ix[ch] for ch in phrase]
        #Compute states for sequence
        for chr in phrase_ix:
            self.excite_rnn(chr)

        # Compute first letter after phrase
        y = np.dot(self.Why, self.hprev) + self.by
        p = np.exp(y/self.temperature) / np.sum(np.exp(y/self.temperature))
        ix = np.random.choice(range(self.vocab_size), p=p.ravel())

        # Keep generating letters:
        sample = self.ix_to_char[ix]
        len = 0
        while True:
            p = self.get_prob(ix)[0]
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            char = self.ix_to_char[ix]
            len += 1
            if char not in ['.', '\n', '!', '?'] or len < minlenght:
                sample += char
            else:
                sample += char
                break
        return phrase+sample


    def save_rnn(self):
        '''
            Saves the RNN parameters to disk
        '''
        np.savez_compressed("saved_rnn.npz", **vars(self))
        self.saves += 1
        print_warning("[I] Model saved to disk.")


    def get_iter(self):
        '''
            Calculates how many training iterations the RNN has had
        :return: Number of training iterations
        '''
        return self.saves * self.max_updates


    def reset_state(self):
        '''
            Resets the RNN state to a zero-vector
        '''
        self.hprev = np.zeros_like(self.hprev)

    def one_hot(self, v):
        '''
            Encodes a character in an one-hot vector
        :param v: Character id
        :return: One-hot encoding of character
        '''
        return np.vstack(np.eye(self.vocab_size)[v])


    def reload_rnn(self, file='saved_rnn.npz'):
        with np.load(open(file)) as a:
            for p in a.files:
                if p not in ['char_to_ix', 'ix_to_char']:
                    self.set_param(p, a[p])


    def set_param(self, param, value):
        '''
                UNSAFE
            Sets a RNN parameter
        :param param: Parameter name
        :param value: New parameter value
        :return: Previous parameter value
        '''
        bkp = None
        try:
            exec("bkp = self.%s" %(param))
            exec("self.%s = value" %(param))
        except AttributeError:
            print "RNN has no parameter named %s. SKIPPING." %param

        return bkp



    def test_sequence(self, init, next, samples=500):
        init_ix = self.char_to_ix[init]
        next_ix = self.char_to_ix[next]

        self.excite_rnn(init_ix)
        hprev_avg = self.hprev
        p = self.get_prob(init_ix)[0]

        chars = np.array(self.ix_to_char.values())
        for i in range(samples):
            # "Reshuffle" RNN state
            self.sample_rnn(chars[np.random.randint(0,len(chars))], 10)

            # Compute state after feeding init_ix
            self.excite_rnn(init_ix)
            hprev_avg = (hprev_avg + self.hprev)/2

        pred = hprev_avg.ravel()*self.Why[next_ix,:]
        ibprev = np.argsort(pred)[::-1][:10]
        avg_wxh = np.mean(self.Wxh[:, init_ix])
        best_weights = [i for i in ibprev if self.Wxh[i, init_ix] > 0 and self.Wxh[i, init_ix] > avg_wxh]


        print_info("Hypothesis with %.2f%% probability" % (p[next_ix]*100))
        nexchar = np.argmax(p)
        genchar = self.ix_to_char[nexchar]
        print_info("\t%s with highest probability (%.2f%%) after %s"
                    %(repr(genchar), p[nexchar]*100, repr(init)))
        if(genchar != next):
            print_info("\t%s with %.2f%% probability"
                       %(repr(next), p[next_ix]*100))
        print "Weights Involved:"
        print "\tWxh at [:,%d]" % init_ix
        print "\tWhy at [%d,:]" % next_ix

        print "Most relevant weighs:"
        print "\tWxh at [[%s], %d]" %(', '.join((str(s) for s in best_weights)), init_ix)
        print "\tWhy at [%d, [%s]]" %(next_ix, ', '.join((str(s) for s in best_weights)))

        return best_weights


    def find_association(self, chr, temperature=1, numsamples=100):
        a = []
        ix = self.char_to_ix[chr]
        chars = self.ix_to_char
        for i in range(numsamples):
            p = self.get_prob(ix)[0]
            ipmax = np.argmax(p)
            a.append(ipmax)
        return self.ix_to_char[max(set(a), key=a.count)]
