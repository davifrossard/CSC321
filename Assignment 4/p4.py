from RNN import *
from utils import print_warning
from os import makedirs
from time import sleep
import datetime, time

# Create results directory
if not path.exists('results/'):
    makedirs('results/')

# Data I/O
data = open('shakespeare_train.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'Data has %d characters, %d unique.' % (data_size, vocab_size)

# Hyper-parameters
hidden_size = 250
seq_length = 25
learning_rate = 1e-1
temperature = 1

# Create model
RNN = RNN_Model(hidden_size, chars, temperature)

# Smooth Out loss
smooth_loss = -np.log(1.0/vocab_size)*seq_length

p = 0
n = RNN.get_iter()
print "Training RNN, hit Ctrl+C to stop."
try:
    while True:
        n += 1

        # Create training set from file
        inputs = data[p:p+seq_length]
        targets = data[p+1:p+seq_length+1]

        # Update RNN with data
        loss = RNN.update_rnn(inputs, targets, learning_rate)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Increment data pointer with wrap-around
        if p+2*seq_length+1 < len(data):
            p += seq_length
        else:
            p = 0
            RNN.reset_state()
            print_warning("[I] Finished pass through file.")

        # Show progress
        if n % 250 == 0:
            print "---------------------------------------------"
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            print "[%s] Iteration: \t\t%d" % (st, n)
            print "\t\t   Loss: \t\t\t%7.4f" % smooth_loss
            print "\t\t   Characters Fed:  %d" % p
            print "\t\t   Sample: \n\n"
            print RNN.sample_rnn(data[p], 200)
            print "---------------------------------------------\n\n"

except KeyboardInterrupt:
    print "Halting training"


#
# ------------------------
#          PART 1
# ------------------------
print_info("\n\n---------------------\n"
           "RUNNING PART 1\n"
           "---------------------\n")
temperatures = [0.01, 0.01, 0.1, 0.5, 0.7, 1, 1.5]
for i in temperatures:
    samples = []
    for j in range(5):
        samples.append(RNN.sample_rnn(data[np.random.randint(0, len(data))], 200, i))
    samples = '\n-----------------------------------------\n\n'.join(s.rstrip() for s in samples)
    print samples
    with open('results/samples_%4f.txt' % i, 'w+') as fl:
        fl.write(samples)


#
# ------------------------
#          PART 2
# ------------------------
print_info("\n\n---------------------\n"
           "RUNNING PART 2\n"
           "---------------------\n")
samples = []
for i in range(10):
    temp = np.random.choice(temperatures[-3::])
    samples.append(RNN.complete_phrase("The answer to life the universe and everything is ", temp))
samples = '\n-----------------------------------------\n\n'.join(s.rstrip() for s in samples)
print samples
with open('results/completion.txt', 'w+') as fl:
    fl.write(samples)


#
# ------------------------
#          PART 3
# ------------------------
print_info("\n\n---------------------\n"
           "RUNNING PART 3\n"
           "---------------------\n")
RNN.reload_rnn('char-rnn-snapshot.npz')
best_weights = RNN.test_sequence(':', '\n')

init_ix = RNN.char_to_ix[':']
end_ix = RNN.char_to_ix['\n']

with open('results/part3_weights.txt', 'w+') as f:
    f.write('Input to State Weights: [%s, %d]'
            'State to Output Weights: [%d, %s]'
            %(best_weights, init_ix,
              end_ix, best_weights))

#
# ------------------------
#          PART 4
# ------------------------
print_info("\n\n---------------------\n"
           "RUNNING PART 4\n"
           "---------------------\n")
RNN.reload_rnn()

associations = []
for char in sorted(chars):
    res = RNN.find_association(char, 0.6)
    association = repr("%s [%2d] -> %s [%2d]" % (char, RNN.char_to_ix[char],
                                               res, RNN.char_to_ix[res]))
    print association
    associations.append(association)

with open('results/part4_associations.txt', 'w+') as f:
    f.write('%s' %  '\n'.join(a for a in associations))

best_weights = RNN.test_sequence('S', ':')

init_ix = RNN.char_to_ix['S']
end_ix = RNN.char_to_ix[':']

with open('results/part3_weights.txt', 'w+') as f:
    f.write('Input to State Weights: [%s, %d]'
            'State to Output Weights: [%d, %s]'
            %(best_weights, init_ix,
              end_ix, best_weights))