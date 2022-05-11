import os
import csv
import random
import argparse
from collections import defaultdict

import jsonlines


def load_data(filepath):
    new_data = []
    with open(filepath, encoding="utf-8") as f:
        data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            new_data.append({
                "text": row[1],
                "label": int(row[0]) - 1,
            })
    return new_data


def balanced_samples(data, size):
    cache = defaultdict(list)
    for d in data:
        cache[d['label']].append(d)
    for l in cache.values():
        random.shuffle(l)

    sample_set = list()
    for label, examples in cache.items():
        if len(examples) < size:
            print(f"number of target samples ({size}) more than "
                            f"the number of examples ({len(examples)}) "
                            f"in the class ({label})")
        for _ in range(min(size, len(examples))):
            sample_set.append(examples.pop())

    return sample_set, sum(cache.values(), [])


def main(args):
    # load data
    in_data = "./data/yelp_full_test.jsonl"
    with jsonlines.open(in_data) as reader:
        full_train_data = list(reader)
    num_labels = len(set(d['label'] for d in full_train_data))
    num_subsample_per_label = int(len(full_train_data) * args.subsample_ratio / num_labels)
    random.seed(args.seed)
    subsampled_data, _ = balanced_samples(full_train_data, num_subsample_per_label)

    print(f"Resulting dataset size: {len(subsampled_data):,}")

    out_data = f'./data/yelp_subsample={args.subsample_ratio}_seed={args.seed}_train.jsonl'
    with jsonlines.open(out_data, 'w') as writer:
        writer.write_all(subsampled_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample_ratio', type=float, required=True)
    args = parser.parse_args()
    main(args)
