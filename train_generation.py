import os
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer


DATA_CACHE_DIR = "./data/.cache"
MODEL_CACHE_DIR = './models/.cache'


def load_tokenized_dataset(raw_dataset, tokenizer, max_len=256):
    def preprocess_function(examples):
        sentiments = examples['label']
        masked_sentences = examples['masked_text']
        prefixes = ["generate negative texts: ",
                    "generate negative texts: ",
                    "generate neutral texts: ",
                    "generate positive texts: ",
                    "generate positive texts: "]

        sentences_with_prefix = []
        for sent, masked in zip(sentiments, masked_sentences):
            sentences_with_prefix.append(prefixes[sent] + masked)

        tokenized_inputs = tokenizer(sentences_with_prefix, max_length=max_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            tokenized_outputs = tokenizer(examples['text'], max_length=max_len, truncation=True)

        model_inputs = {}
        model_inputs["attention_mask"] = tokenized_inputs["attention_mask"]
        model_inputs['input_ids'] = tokenized_inputs['input_ids']
        model_inputs["labels"] = tokenized_outputs['input_ids']
        return model_inputs

    tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)
    print(tokenized_datasets.cache_files)

    return tokenized_datasets


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def main(args, training_args):
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=MODEL_CACHE_DIR)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=MODEL_CACHE_DIR) # ==> SentencePiece
    # roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base') # ==> BPE, XLNET # WordPiece
    gen_model.config.max_length = 512

    # add roberta's mask_token to the t5 tokenizer and increase the embedding size.
    gen_model.resize_token_embeddings(len(gen_tokenizer))

    yelp_dataset = load_dataset('json', data_files=args.in_data, cache_dir=DATA_CACHE_DIR, split='train')
    yelp_dataset = yelp_dataset.train_test_split(test_size=0.1, seed=training_args.seed)
    tokenized_yelp_dataset = load_tokenized_dataset(yelp_dataset, tokenizer=gen_tokenizer, max_len=512)


    metric = load_metric("sacrebleu", DATA_CACHE_DIR)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = gen_tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, gen_tokenizer.pad_token_id)
        decoded_labels = gen_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != gen_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Convert eval_every_n_epochs to eval_steps
    train_data_size = len(tokenized_yelp_dataset['train'])
    print("Train dataset size: ", train_data_size)
    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    print("Total Train batch size: ", total_train_batch_size)
    training_args.eval_steps = int(train_data_size / total_train_batch_size * args.eval_every_n_epochs)
    training_args.save_steps = training_args.eval_steps
    print("Eval every {} epochs = every {} steps".format(args.eval_every_n_epochs, training_args.eval_steps))

    trainer = Seq2SeqTrainer(
        gen_model,
        training_args,
        train_dataset=tokenized_yelp_dataset["train"].remove_columns("label"),
        eval_dataset=tokenized_yelp_dataset["test"].remove_columns("label"),
        tokenizer=gen_tokenizer,
        data_collator=DataCollatorForSeq2Seq(gen_tokenizer, padding=True),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    # create symbole link of best chekpoint for easy access
    best_model_ckpt_path = Path(trainer.state.best_model_checkpoint).resolve()
    os.symlink(best_model_ckpt_path, best_model_ckpt_path.parent / 'checkpoint-best')


if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--eval_every_n_epochs', type=float, default=0.5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    training_args, additional_args = parser.parse_args_into_dataclasses()
    main(additional_args, training_args)
