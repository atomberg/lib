import numpy as np
from tqdm import trange


def iterate_minibatches(X, Y, batchsize, shuffle=False, progress=False):
    assert len(X) == len(Y)
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in (trange if progress else range)(0, len(X) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], Y[excerpt]
