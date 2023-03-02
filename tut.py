'''
based on https://www.tensorflow.org/text/tutorials/text_generation
'''
import tensorflow as tf
import numpy as np
import os
import time

# The Model
class textModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

# Generate text from model one char at a time
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states


# take ids and turn them back into readable str
def text_from_ids(ids):
    text = tf.strings.reduce_join(chars_from_ids(ids), axis=-1) # research axis=-1
    return text

# used for training
def split_input(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# generate and print out text from model
def generate_and_print(constant, one_step_model):
    start = time.time()
    states = None
    next_char = tf.constant([constant])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)


if __name__ == "__main__":
    text = open('puss_2.txt', 'rb').read().decode(encoding='UTF-8')
    print(type(text))
    

    # ----------------------------------PART 1: Initialize some variables----------------------------------
    # vocab
    # chars
    # ids


    # get ids from chars
    # get chars from ids
        # get text from ids - print out readable text


    # ----------------------------------PART 2: Create the dataset----------------------------------
    # convert text vector into a stream of character indices using ids
    

    # using 'batch' method we can convert individual chars to sequences of any length


    # need a paired dataset - input (seq), label(seq) pairs
    # (used for training/testing)
    # split_input()


    # now split the dataset into pairs 


    # create training batches
    # goal: shuffle data, pack into batches
    BATCH_SIZE = 64 
    BUFFER_SIZE = 10000 # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


    # ----------------------------------PART 3: Build the bare bones model----------------------------------
    # model parameters to be used later
    vocab_size = len(ids_from_chars.get_vocabulary())
    embedding_dim = 256
    rnn_units = 1024

    model = textModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    
    # test the (untrained) model


    # let's see what our predictions look like
    

    # we need a loss function to see the 'accuracy' of the model
   

    # ----------------------------------PART 4: Train the model----------------------------------
    # add optimizer, loss function
    

    # configure checkpoints for training

    
    # run the training
    

    # save the model
    tf.saved_model.save(one_step_model, 'one_step')

    # ----------------------------------PART 5: Generate Text----------------------------------
    # head to test.py
    generate_and_print("P", one_step_model)


'''
improving model tips:
-run more epochs
-adjust batch, buffer size (trial and error)
-change start char -> string
-add another RNN layer
-adjust temperature parameter 
  -higher temp = more diversity, more silliness and mistakes
  -lower temp = more accurate, less 'newness'
    -most people just use 1
'''