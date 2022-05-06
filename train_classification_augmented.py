import os

import numpy as np
from datasets import load_metric, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, \
    TrainingArguments


DATA_DIR = './data/text_gen/'
MODEL_DIR = './models/text_gen/'


def main():
    yelp_dataset = load_dataset("./my_yelp_review_full.py", cache_dir=DATA_DIR)

    augmented_dataset_4 = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_4")
    augmented_dataset_5 = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_5")
    augmented_dataset_6 = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_6")
    augmented_dataset_7 = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_7")
    # augmented_dataset_4_opp = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_4_opp")
    # augmented_dataset_5_opp = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_5_opp")
    # augmented_dataset_6_opp = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_6_opp")
    # augmented_dataset_7_opp = load_dataset("./generated_yelp_5.py", cache_dir=DATA_DIR, name="generated_yelp_7_opp")

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('./models/text_gen/yelp_full/roberta_tc_results/checkpoint-53000/')

    metric = load_metric("accuracy")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    print('Tokenizing dataset...')
    def preprocess_data(examples):
        return tokenizer(examples['text'], truncation=True)
    tokenized_dataset = yelp_dataset.map(preprocess_data, batched=True)
    tokenized_augmented_4 = augmented_dataset_4.map(preprocess_data, batched=True)
    tokenized_augmented_5 = augmented_dataset_5.map(preprocess_data, batched=True)
    tokenized_augmented_6 = augmented_dataset_6.map(preprocess_data, batched=True)
    tokenized_augmented_7 = augmented_dataset_7.map(preprocess_data, batched=True)
    # tokenized_augmented_4_opp = augmented_dataset_4_opp.map(preprocess_data, batched=True)
    # tokenized_augmented_5_opp = augmented_dataset_5_opp.map(preprocess_data, batched=True)
    # tokenized_augmented_6_opp = augmented_dataset_6_opp.map(preprocess_data, batched=True)
    # tokenized_augmented_7_opp = augmented_dataset_7_opp.map(preprocess_data, batched=True)

    tokenized_augmented_dataset = concatenate_datasets([tokenized_augmented_4['train'],
                                                                tokenized_augmented_5['train'],
                                                                tokenized_augmented_6['train'],
                                                                tokenized_augmented_7['train']])

    train_dataset = concatenate_datasets([tokenized_augmented_dataset, tokenized_dataset['train']])

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "yelp_full", "roberta_tc_augmented_all"),
        evaluation_strategy = "steps",
        eval_steps=1000,
        learning_rate=1e-6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        num_train_epochs=30,
        weight_decay=0.01,
        save_total_limit=15,
    )

    print('Running training...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
