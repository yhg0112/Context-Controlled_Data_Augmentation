import os

import random
import jsonlines

import numpy as np
from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, HfArgumentParser, \
    Trainer, TrainingArguments

from subsample_yelp_data import balanced_samples


DATA_CACHE_DIR = "./data/.cache"


def main(args, training_args):
    # load data
    full_train_data_path = './data/yelp_full_train.jsonl'
    if args.subsample_ratio is not None:
        train_data_path = f'./data/yelp_subsample={args.subsample_ratio}_seed={training_args.seed}_train.jsonl'
        if not os.path.exists(train_data_path):
            print(f"Subsampled dataset with ratio {args.subsample_ratio} file not found.. Creating with seed {training_args.seed}!")
            with jsonlines.open(full_train_data_path) as reader:
                full_train_data = list(reader)
            num_labels = len(set(d['label'] for d in full_train_data))
            num_subsample_per_label = int(len(full_train_data) * args.subsample_ratio / num_labels)

            random.seed(training_args.seed)
            subsampled_data, _ = balanced_samples(full_train_data, num_subsample_per_label)
            print(f"Resulting dataset size: {len(subsampled_data):,}")

            with jsonlines.open(train_data_path, 'w') as writer:
                writer.write_all(subsampled_data)
    else:
        train_data_path = full_train_data_path
    data_files = {
        'train': train_data_path,
        'test': './data/yelp_full_test.jsonl'
        }

    yelp_dataset = load_dataset('json', data_files=data_files, cache_dir=DATA_CACHE_DIR)

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=5)

    print('Tokenizing dataset...')
    def preprocess_data(examples):
        return tokenizer(examples['text'], truncation=True)
    tokenized_dataset = yelp_dataset.map(preprocess_data, batched=True)

    metric = load_metric("accuracy")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    print('Running training...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    parser.add_argument('--subsample_ratio', type=float, default=None)
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    training_args, additional_args = parser.parse_args_into_dataclasses()
    main(additional_args, training_args)
