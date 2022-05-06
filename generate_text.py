import os
import json

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration


DATA_DIR = './data/text_gen/'
MODEL_DIR = './models/text_gen/'


def main():
    device = torch.device("cuda")
    yelp_dataset_7 = load_dataset("./masked_yelp_5.py", cache_dir=DATA_DIR, name="masked_yelp_7")

    model_name_or_path = "./models/text_gen/masked_yelp_full/t5-v1_1-small_text_generation_ft_p7/checkpoint-50500/"

    gen_tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    gen_model_7 = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    gen_model_7.config.max_length = 512

    prefixes = ["generate negative texts: ",
                "generate negative texts: ",
                "generate neutral texts: ",
                "generate positive texts: ",
                "generate positive texts: "]

    def load_tokenized_dataset(raw_dataset, tokenizer, max_len=256):

        def preprocess_function(examples):
            sentiments = examples['label']
            masked_sentences = examples['masked_text']
            # prefixes = ["generate 1 star texts: ",
            #             "generate 2 star texts: ",
            #             "generate 3 star texts: ",
            #             "generate 4 star texts: ",
            #             "generate 5 star texts: "]

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


    tokenized_yelp_dataset_7 = load_tokenized_dataset(yelp_dataset_7, tokenizer=gen_tokenizer, max_len=512)

    masked_texts = tokenized_yelp_dataset_7["test"]['masked_text']
    original_label = tokenized_yelp_dataset_7["test"]['label']


    batch_size = 32
    generated_outputs_7 = []
    generated_labels_7 = []
    with torch.no_grad():
        for i in tqdm(range((len(masked_texts) // batch_size) + 1), desc='generate'):
            batch_masked_texts = masked_texts[i*batch_size:(i+1)*batch_size]
            batch_label = original_label[i*batch_size:(i+1)*batch_size]

            # add prefix
            for j, (text, label) in enumerate(zip(batch_masked_texts, batch_label)):
                batch_masked_texts[j] = prefixes[label] + batch_masked_texts[j]
            encoded_texts = gen_tokenizer(batch_masked_texts,
                                        padding=True,
                                        return_tensors="pt").to(device)

            outs = gen_model_7.generate(**encoded_texts, num_beams=4, typical_p=0.2, do_sample=True)
            batch_decoded = gen_tokenizer.batch_decode(outs, skip_special_tokens=True)
            generated_outputs_7 += batch_decoded
            generated_labels_7 += batch_label

    batch_size = 32
    generated_outputs_7_opp = []
    generated_labels_7_opp = []
    with torch.no_grad():
        for i in tqdm(range((len(masked_texts) // batch_size) + 1), desc='generate'):
            batch_masked_texts = masked_texts[i*batch_size:(i+1)*batch_size]
            batch_label = original_label[i*batch_size:(i+1)*batch_size]
            batch_label_opp = []

            # add prefix
            for j, (text, label) in enumerate(zip(batch_masked_texts, batch_label)):
                if label < 2:
                    opp_label = 4
                elif label == 2:
                    opp_label = 2
                else:
                    opp_label = 0
                batch_label_opp.append(opp_label)
                batch_masked_texts[j] = prefixes[opp_label] + batch_masked_texts[j]
            encoded_texts = gen_tokenizer(batch_masked_texts,
                                        padding=True,
                                        return_tensors="pt").to(device)

            outs = gen_model_7.generate(**encoded_texts, num_beams=4, typical_p=0.2, do_sample=True)
            batch_decoded = gen_tokenizer.batch_decode(outs, skip_special_tokens=True)
            generated_outputs_7_opp += batch_decoded
            generated_labels_7_opp += batch_label


    # save the generated texts:
    generated_dataset_7 = []
    for i in range(len(generated_outputs_7)):
        ex = {}
        ex['text'] = generated_outputs_7[i]
        ex['label'] = generated_labels_7[i]
        generated_dataset_7.append(ex)

    generated_dataset_7_opp = []
    for i in range(len(generated_outputs_7_opp)):
        ex = {}
        ex['text'] = generated_outputs_7_opp[i]
        ex['label'] = generated_labels_7_opp[i]
        generated_dataset_7_opp.append(ex)

    with open(os.path.join(DATA_DIR, "generated_yelp_7_again.json"), 'w') as json_file:
        json.dump(generated_dataset_7, json_file)
    with open(os.path.join(DATA_DIR, "generated_yelp_7_opp_again.json"), 'w') as json_file:
        json.dump(generated_dataset_7_opp, json_file)


if __name__ == "__main__":
    main()
