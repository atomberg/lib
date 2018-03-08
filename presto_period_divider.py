import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm


def get_trips(filename):
    T = pd.read_csv(filename, dayfirst=True, parse_dates=['Date'])
    T = T['Date'].map(lambda d: d.date()).map(datetime.date.toordinal)
    trips = np.zeros_like(list(range(T.min(), T.max() + 1)))
    for i in T:
        trips[i - T.min()] += 1
    return trips


trips = get_trips('~/Downloads/TUR_2018_06495883_024.csv')
answers = {}


def optimise(j):
    if j in answers:
        return answers[j]
    if j < 32:
        if trips[:j].sum() > 31:
            answers[j] = [0], trips[0:j].sum()
        else:
            answers[j] = [], -1
    else:
        i = j - 31
        max_score, answer = -1, (None, None)
        while trips[i:j].sum() > 31:
            previous, score = optimise(i)
            if score + trips[i:j].sum() > max_score:
                max_score, answer = score + trips[i:j].sum(), previous + [i]
            i += 1
        if max_score < 0:
            answers[j] = [], -1
        else:
            answers[j] = answer, max_score
    return answers[j]


for i in tqdm(range(len(trips))):
    optimise(i)

periods, score = optimise(len(trips))
periods.append(len(trips))
for i, j in zip(periods, periods[1:]):
    start = datetime.date.fromordinal(736331 + i)
    stop = datetime.date.fromordinal(736331 + j - 1)
    print(f"{start} to {stop} -- {j-i} days & {trips[i:j].sum()} trips")
