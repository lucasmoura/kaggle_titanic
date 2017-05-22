import random

import numpy as np


def create_batches(data, batch_size):
    random.shuffle(data)
    batches = [data[offset:offset+batch_size]
               for offset in range(0, len(data), batch_size)]
    return batches


def unify_batch(batch):
    data_batch, prediction_batch = batch[0]

    for data, prediction in batch[1:]:
        data_batch = np.concatenate((data_batch, data), axis=1)
        prediction_batch = np.concatenate(
            (prediction_batch, prediction), axis=1)

    return(data_batch, prediction_batch.T)
