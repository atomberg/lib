import argparse
import datetime
import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm
from typing import Mapping, List, Tuple


def main(csvfile: str):

    csv = pd.read_csv(csvfile, dayfirst=True, parse_dates=['Date'])

    trip_counts = Counter(csv['Date'].map(lambda d: d.date().toordinal()))
    days, counts = zip(*trip_counts.items())
    trips = np.zeros(shape=(max(trip_counts) - min(trip_counts) + 1,), dtype=int)
    trips[np.array(days) - min(trip_counts)] = np.array(counts)

    answers: Mapping[int, Tuple[List[int], int]] = {}

    def optimization_step(j: int) -> Tuple[List[int], int]:
        """Find the optimal solution for all trips from day 0 to day j."""
        if j not in answers:
            # Base case: we can fit all days into one period, so let's be greedy
            if j < 32:
                answers[j] = ([0], trips[:j].sum() if trips[:j].sum() > 31 else 0)

            # Recursive case: opt[j] = max(period[i:j] + opt[i] for j - 31 <= i < j)
            else:
                best_score = -1
                for i in range(j - 31, j):
                    prev_answer, prev_score = optimization_step(i)
                    score = prev_score + (trips[i:j].sum() if trips[i:j].sum() > 31 else 0)
                    if score > best_score:
                        best_score, best_answer = score, prev_answer + [i]
                answers[j] = best_answer, best_score

        return answers[j]

    for i in tqdm(range(len(trips) + 1), unit_scale=True, leave=False, desc='Optimizing periods'):
        optimization_step(i)

    periods, trips_used = answers[max(answers)]
    for i, j in zip(periods, periods[1:] + [len(trips)]):
        start = datetime.date.fromordinal(min(trip_counts) + i)
        stop = datetime.date.fromordinal(min(trip_counts) + j - 1)
        score = trips[i:j].sum() if trips[i:j].sum() > 31 else 0
        print(f"{start} to {stop} -- {j-i} days & {score} / {trips[i:j].sum()} trips used")
    print(f"--- Total trips used: {trips_used} / {trips.sum()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("trips", help="Exported trip summary in CSV format from prestocard.ca")

    args = parser.parse_args()
    main(args.trips)
