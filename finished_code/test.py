import tensorflow as tf
import numpy as np
import time

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


one_step = tf.saved_model.load('one_step')
generate_and_print("Well, ", one_step)
