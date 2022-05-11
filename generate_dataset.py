import argparse
import jsonlines

import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATA_CACHE_DIR = "./data/.cache"


def generate(model, tokenizer, dataset, batch_size, prefixes, flip_label=False):
    loader = DataLoader(dataset, batch_size=batch_size)

    generated_texts = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if flip_label:
                batch_labels = []
                for label in batch['label']:
                    if label < 2:
                        label = 4
                    elif label == 2:
                        label = 2
                    else:
                        label = 0
                    batch_labels.append(label)
            else:
                batch_labels = batch['label'].tolist()
            prefix_added_batch_masked_texts = [prefixes[label] + masked_text for label, masked_text in zip(batch_labels, batch['masked_text'])]

            encoded_texts = tokenizer(prefix_added_batch_masked_texts, padding=True, return_tensors="pt").to(model.device)
            outs = model.generate(**encoded_texts, num_beams=4, typical_p=0.2, do_sample=True)

            generated_texts.extend(tokenizer.batch_decode(outs, skip_special_tokens=True))
            labels.extend(batch_labels)

    return [{ 'text': text, 'label': label } for text, label in zip(generated_texts, labels) ]


def main(args):
    device = torch.device("cuda")

    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

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

    masked_dataset = load_dataset('json', data_files=args.in_data, cache_dir=DATA_CACHE_DIR, split='train')
    masked_dataset = masked_dataset.train_test_split(test_size=0.1, seed=args.seed)
    # tokenized_masked_test_dataset = load_tokenized_dataset(masked_dataset['test'], tokenizer=gen_tokenizer, max_len=args.max_len)

    generated_dataset = generate(gen_model, gen_tokenizer, masked_dataset['test'], args.batch_size, prefixes, flip_label=False)
    with jsonlines.open(args.out_data, 'w') as writer:
        writer.write_all(generated_dataset)
    generated_opp_dataset = generate(gen_model, gen_tokenizer, masked_dataset['test'], args.batch_size, prefixes, flip_label=True)
    with jsonlines.open(args.out_opp_data, 'w') as writer:
        writer.write_all(generated_opp_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    parser.add_argument('--out_opp_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)
    main(args)
