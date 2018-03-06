import numpy as np
import pandas as pd
import datetime


def get_trips(filename):
    T = pd.read_csv(filename, dayfirst=True, parse_dates=['Date'])
    T = T['Date'].map(lambda d: d.date()).map(datetime.date.toordinal)
    trips = np.zeros_like(list(range(T.min(), T.max() + 1)))
    for i in T:
        trips[i - T.min()] += 1
    return trips


trips = get_trips('Downloads/TUR_2018_06495883_024.csv')
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
        i = j - 32
        max_score, answer = -1, (None, None)
        while trips[i:j].sum() > 31:
            previous, score = optimise(i)
            if score > max_score:
                max_score, answer = score, (i, previous)
            i += 1
        if max_score < 0:
            answers[j] = [], -1
        else:
            answers[j] = answer[1] + [answer[0]], max_score + trips[answer[0]:j].sum()
    return answers[j]


periods, score = optimise(len(trips))
periods.append(len(trips))
for i, j in zip(periods, periods[1:]):
    start = datetime.date.fromordinal(736331 + i)
    stop = datetime.date.fromordinal(736331 + j - 1)
    print(f"{start} to {stop} -- {j-i} days & {trips[i:j].sum()} trips")
