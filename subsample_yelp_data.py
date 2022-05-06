import csv
import random
import os
from collections import defaultdict


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


def main():
    train_data_path = "./data/text_gen/yelp/yelp_review_full_csv/train.csv"
    seed = 42
    subsample_ratio = 0.01

    train_data = load_data(train_data_path)
    num_labels = len(set(d['label'] for d in train_data))
    num_subsample_per_label = int(len(train_data) * subsample_ratio / num_labels)

    random.seed(seed)
    subsampled_data, _ = balanced_samples(train_data, num_subsample_per_label)

    out_dir = f"./data/text_gen/yelp/yelp_review_subsample={subsample_ratio}_seed={seed}_csv"

    print(f"Resulting dataset size: {len(subsampled_data):,}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'train.csv'), 'w', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for d in subsampled_data:
            writer.writerow((str(d['label']+1), d['text']))


if __name__ == "__main__":
    main()
