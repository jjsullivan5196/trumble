import numpy as np
import tensorflow as tf
import trumble as tb

tweets = tb.collect_tweets(download = False)

encode = {char:ind for ind, char in enumerate(np.unique(tweets))}
decode = {ind:char for ind, char in enumerate(np.unique(tweets))}

char_count = len(np.unique(tweets)) # Total number of characters / output size

X = [x[:-1] for x in tweets]
y = [y[1:] for y in tweets]

seq_length = np.array([len(''.join(tweet)) for tweet in X])
X_numeric = [[encode[alpha] for alpha in tweet] for tweet in X]
y_numeric = [[encode[alpha] for alpha in tweet] for tweet in y]

# Training set with sequence length
X_train = np.array(X_numeric[:20000])
y_train = np.array(y_numeric[:20000])
seq_length_train = seq_length[:20000]

y_train = y_train.flatten()

# Tensorflow

tf.reset_default_graph()

n_steps = 139 # One time step for each character
n_inputs = 1 # One character at a time
n_neurons = 100
n_outputs = char_count
n_layers = 1 # Single layer to make experimentation faster

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(X_train)//batch_size):
            X_batch = X_train[iteration*batch_size:iteration*batch_size+batch_size]
            y_batch = y_train[iteration*batch_size:iteration*batch_size+batch_size]
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("Epoch", epoch, "Train accuracy =", acc_train)
