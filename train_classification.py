import os
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, HfArgumentParser, \
    Trainer, TrainingArguments, EarlyStoppingCallback


DATA_CACHE_DIR = "./data/.cache"


def main(args, training_args):
    yelp_dataset = load_dataset('json', data_files={
        'train': args.train_data_paths,
        'test': args.test_data_path,
    }, cache_dir=DATA_CACHE_DIR)

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

    # create symbole link of best chekpoint for easy access
    best_model_ckpt_path = Path(trainer.state.best_model_checkpoint).resolve()
    os.symlink(best_model_ckpt_path, best_model_ckpt_path.parent / 'checkpoint-best')

    # final evaluation on full test data
    trainer.evaluate(tokenized_dataset['test'], metric_key_prefix="full_eval")


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    parser.add_argument('--train_data_paths', nargs='+', default=[])
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--subsample_ratio', type=float, default=None)
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--eval_every_n_epochs', type=float, default=0.5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--subsample_eval_set_size', type=int, default=None)
    training_args, additional_args = parser.parse_args_into_dataclasses()
    main(additional_args, training_args)
