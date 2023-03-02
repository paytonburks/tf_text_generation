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


if __name__ == "__main__":
    text = open('puss_2.txt', 'rb').read().decode(encoding='UTF-8')
    print(type(text))
    

    # ----------------------------------PART 1: Initialize some variables----------------------------------
    # show each unique char
    vocab = sorted(set(text))
    print(vocab)

    # vectorize
    # split the text into tokens
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    print(chars)

    # now a function to convert each character into a numeric ID
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None # https://www.tensorflow.org/guide/keras/masking_and_padding
    )
    ids = ids_from_chars(chars)
    print(ids)

    # we are going to need a way to convert the IDs back to char vectors
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )
    chars = chars_from_ids(ids)
    print(chars)


    # ----------------------------------PART 2: Create the dataset----------------------------------
    # convert text vector into a stream of character indices using ids
    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)

    # example use
    for ids in ids_dataset.take(5):
        print(chars_from_ids(ids).numpy().decode('utf-8')) # see that this is char by char
    
    # using 'batch' method we can convert individual chars to sequences of any length
    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    # example use
    for seq in sequences.take(1):
        print(chars_from_ids(seq))
        print(text_from_ids(seq).numpy())

    # need a paired dataset - input (seq), label(seq) pairs
    # (used for training/testing)
    # split_input()
    print(split_input(list("Horton Hears a Who!")))

    # now split the dataset into pairs 
    dataset = sequences.map(split_input)

    # use example of dataset
    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(input_example).numpy())
        print("Target:", text_from_ids(target_example).numpy())

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
    for input_ex, target_ex in dataset.take(1):
        ex_pred = model(input_ex)
        print(ex_pred.shape)
    model.summary()

    # let's see what our predictions look like
    sampled_indices = tf.random.categorical(ex_pred[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print("Next Char Predictions:", text_from_ids(sampled_indices).numpy())

    # we need a loss function to see the 'accuracy' of the model
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    ex_pred_loss = loss(target_ex, ex_pred)
    print("Example Mean loss:", ex_pred_loss)

    # note that our exp(loss) should be approx equal to vocab size
    print(tf.exp(ex_pred_loss).numpy())
    

    # ----------------------------------PART 4: Train the model----------------------------------
    # add optimizer, loss function
    model.compile(optimizer='adam', loss=loss)

    # configure checkpoints for training
    # DIR
    checkpoint_dir = './training_checkpoints'
    # Name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    # run the training
    EPOCHS = 30
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    # save the model
    tf.saved_model.save(one_step_model, 'one_step')


    # ----------------------------------PART 5: Generate Text----------------------------------
    start = time.time()
    states = None
    next_char = tf.constant(['PUSS:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)

    
