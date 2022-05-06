import os
import numpy as np

from transformers import AutoTokenizer, T5TokenizerFast, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets


DATA_DIR = './data/text_gen/'
MODEL_DIR = './models/text_gen/'


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


def main():
    batch_size = 16
    model_name = 't5-v1_1-small_text_generation_ft_p7'
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "masked_yelp_full", model_name),
        evaluation_strategy="steps",
        eval_steps=10000,
        learning_rate=3e-4,
        warmup_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=15,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
        metric_for_best_model='bleu'
    )

    gen_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small", cache_dir=MODEL_DIR)
    gen_tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-small", cache_dir=MODEL_DIR) # ==> SentencePiece
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base') # ==> BPE, XLNET # WordPiece
    gen_model.config.max_length = 512


    # add roberta's mask_toekn to the t5 tokenizer and increase the embedding size.
    mask_token_dict = {'mask_token': roberta_tokenizer.mask_token}
    num_added_toks = gen_tokenizer.add_special_tokens(mask_token_dict)
    gen_model.resize_token_embeddings(len(gen_tokenizer))


    yelp_dataset_7 = datasets.load_dataset("./masked_yelp_5.py", cache_dir=DATA_DIR, name="masked_yelp_7")
    tokenized_yelp_dataset_7 = load_tokenized_dataset(yelp_dataset_7, tokenizer=gen_tokenizer, max_len=512)


    data_collator = DataCollatorForSeq2Seq(gen_tokenizer, padding=True)

    metric = datasets.load_metric("sacrebleu", DATA_DIR)

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


    trainer = Seq2SeqTrainer(
        gen_model,
        args,
        train_dataset=tokenized_yelp_dataset_7["train"].remove_columns("label"),
        eval_dataset=tokenized_yelp_dataset_7["test"].remove_columns("label"),
        data_collator=data_collator,
        tokenizer=gen_tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    main()
