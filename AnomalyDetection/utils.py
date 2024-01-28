import numpy as np


def create_sequences(values, seq_len: int):
    output = []
    for i in range(len(values) - seq_len + 1):
        output.append(values[i:(i + seq_len)])
    return np.stack(output)