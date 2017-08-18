import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_file = '../data/creditcard.csv'


def load_dataset(upsample_to=0.125):
    df = pd.read_csv(data_file)

    pos = df[df['Class'] == 1]
    if upsample_to:
        reps = upsample_to / (1 - upsample_to) * (len(df) + len(pos)) / len(pos)
        df = np.concatenate([df, np.tile(pos.values, (int(reps), 1))])
    else:
        df = df.values
    X, Y = df[:, :-1], np.array(df[:, -1], dtype=np.int32)

    # Split the ndarray into training/validation/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=10**4)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
