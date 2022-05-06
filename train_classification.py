import os

import numpy as np
from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, \
    TrainingArguments


DATA_DIR = './data/text_gen/'
MODEL_DIR = './models/text_gen/'


def main():
    yelp_dataset = load_dataset("./my_yelp_review_subsample.py", cache_dir=DATA_DIR)
    model_name_or_path = 'roberta-base'

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=5)

    print('Tokenizing dataset...')
    def preprocess_data(examples):
        return tokenizer(examples['text'], truncation=True)
    tokenized_dataset = yelp_dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        # output_dir=os.path.join(MODEL_DIR, "yelp_full", "roberta_tc_results"),
        output_dir=os.path.join(MODEL_DIR, "yelp_subsample=0.01_seed=42", "roberta_tc_results"),
        evaluation_strategy = "steps",
        eval_steps=1000,
        learning_rate=2e-5,
        fp16=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        num_train_epochs=30,
        weight_decay=0.01,
        save_total_limit=15,
    )

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
    main()
