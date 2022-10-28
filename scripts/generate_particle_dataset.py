"""Script to convert a directory of CSV files to one pickle.

Converts all the CSV files into a list of lists where each CSV file
is an element of the outer list and each inner element is a list of
values converted to floats.
"""


import pickle
import numpy as np
import csv
import argparse
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument(
    "dir",
    help="Directory containing CSV files",
    type=str
)

args = ap.parse_args()

files = pathlib.Path(args.dir).glob("*.csv")

data = []

for file in files:
    with open(file) as f:
        reader = csv.reader(f, delimiter=',')
        d = []
        for row in reader:
            d.append([float(r) for r in row])
    data.append(d)


if len(data) == 0:
    raise RuntimeError(f"Didn't find any CSV files in {args.dir}")

with open(pathlib.Path(args.dir)/"database.pkl", 'wb') as f:
    pickle.dump(data, f)
