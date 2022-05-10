import os

import random
import jsonlines

import numpy as np
from numpy.random import default_rng
from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, HfArgumentParser, \
    Trainer, TrainingArguments, EarlyStoppingCallback

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

    # Convert eval_every_n_epochs to eval_steps
    train_data_size = len(yelp_dataset['train'])
    print("Train dataset size: ", train_data_size)
    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    print("Total Train batch size: ", total_train_batch_size)
    training_args.eval_steps = int(train_data_size / total_train_batch_size * args.eval_every_n_epochs)
    training_args.save_steps = training_args.eval_steps
    print("Eval every {} epochs = every {} steps".format(args.eval_every_n_epochs, training_args.eval_steps))

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

    # subsample test set for evaluation
    if args.subsample_eval_set_size is not None:
        rng = default_rng(seed=training_args.seed)
        subsample_indices = rng.choice(len(tokenized_dataset['test']), size=args.subsample_eval_set_size, replace=False)
        eval_dataset = tokenized_dataset['test'].select(subsample_indices)
    else:
        eval_dataset = tokenized_dataset['test']

    print('Running training...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.evaluate(tokenized_dataset['test'], "")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    parser.add_argument('--subsample_ratio', type=float, default=None)
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--eval_every_n_epochs', type=float, default=0.5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--subsample_eval_set_size', type=int, default=None)
    training_args, additional_args = parser.parse_args_into_dataclasses()
    main(additional_args, training_args)
