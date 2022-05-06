import argparse
import jsonlines

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATA_CACHE_DIR = "./data/.cache"


def main(args):
    device = torch.device("cuda")

    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)
    # gen_model.config.max_length = 512

    prefixes = ["generate negative texts: ",
                "generate negative texts: ",
                "generate neutral texts: ",
                "generate positive texts: ",
                "generate positive texts: "]
    # prefixes = ["generate 1 star texts: ",
    #             "generate 2 star texts: ",
    #             "generate 3 star texts: ",
    #             "generate 4 star texts: ",
    #             "generate 5 star texts: "]

    def load_tokenized_dataset(raw_dataset, tokenizer, max_len):

        def preprocess_function(examples):
            sentiments = examples['label']
            masked_sentences = examples['masked_text']

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

    yelp_dataset = load_dataset('json', data_files=args.in_data, cache_dir=DATA_CACHE_DIR)
    tokenized_yelp_dataset = load_tokenized_dataset(yelp_dataset, tokenizer=gen_tokenizer, max_len=args.max_len)

    masked_texts = tokenized_yelp_dataset["train"]['masked_text']
    original_label = tokenized_yelp_dataset["train"]['label']

    generated_outputs = []
    generated_labels = []
    with torch.no_grad():
        for i in tqdm(range((len(masked_texts) // args.batch_size) + 1), desc='generate'):
            batch_masked_texts = masked_texts[i*args.batch_size:(i+1)*args.batch_size]
            batch_label = original_label[i*args.batch_size:(i+1)*args.batch_size]

            # add prefix
            for j, label in enumerate(batch_label):
                if args.flip_label:
                    if label < 2:
                        label = 4
                    elif label == 2:
                        label = 2
                    else:
                        label = 0
                    batch_label[j] = label
                batch_masked_texts[j] = prefixes[label] + batch_masked_texts[j]
            encoded_texts = gen_tokenizer(batch_masked_texts,
                                        padding=True,
                                        return_tensors="pt").to(device)

            outs = gen_model.generate(**encoded_texts, num_beams=4, typical_p=0.2, do_sample=True)
            batch_decoded = gen_tokenizer.batch_decode(outs, skip_special_tokens=True)
            generated_outputs += batch_decoded
            generated_labels += batch_label

    # save the generated texts:
    generated_dataset = []
    for i in range(len(generated_outputs)):
        ex = {
            'text': generated_outputs[i],
            'label': generated_labels[i],
        }
        generated_dataset.append(ex)

    with jsonlines.open(args.out_data, 'w') as writer:
        writer.write_all(generated_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--flip_label', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)
