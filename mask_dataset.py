import os
import json
from pathlib import Path

import torch
import datasets
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification


softmax = torch.nn.Softmax(dim=-1)
def entropy(p):
    return torch.sum(-1*p*torch.log(p), dim=-1)

def sent_tokenize_and_strip(sentences):
    sentences = sentences.replace("\\n\\n", " ")
    return [s.strip() for s in sent_tokenize(sentences)]


DATA_DIR = './data/text_gen/'
MODEL_DIR = Path('./models/text_gen/')


def main():
    device = torch.device("cuda")

    print('Loading model...')
    # model_name_or_path = 'roberta-base'
    # model_name_or_path = MODEL_DIR / 'yelp_full/roberta_tc_results/checkpoint-128000/'
    model_name_or_path = MODEL_DIR / 'yelp_full/roberta_tc_results/checkpoint-1500/'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)

    def embed(sentences):
        max_len=512
        with torch.no_grad():
            tokenized = tokenizer(sentences, padding=True, max_length=max_len, return_tensors='pt').to(device)
            rslt = sentiment_model(tokenized['input_ids'][:,:max_len])
            return softmax(rslt['logits'])

    yelp_dataset = datasets.load_dataset("./my_yelp_review_full.py", cache_dir=DATA_DIR)

    def mask_sentences_random(sentences, tokenizer, n_sentences=10, p=0.3):
        tokenized_sentences = tokenizer(sentences)['input_ids']
        masked_sentences = []

        def merge_mask_token(sentence):
            to_be_replaced = tokenizer.mask_token + '' + tokenizer.mask_token
            merged_sentence = sentence.replace(to_be_replaced, tokenizer.mask_token)
            if sentence != merged_sentence:
                merged_sentence = merge_mask_token(merged_sentence)
            return merged_sentence

        for tokenized_sentence in tokenized_sentences:
            masked_sentence = np.array([tokenized_sentence] * n_sentences)
            mask_array = np.random.binomial(1, p, size=(masked_sentence.shape[0], masked_sentence.shape[1]-2))
            mask_array = np.concatenate((np.zeros(shape=(masked_sentence.shape[0],1)),
                                        mask_array,
                                        np.zeros(shape=(masked_sentence.shape[0],1))),
                                        axis=1)
            masked_sentence = np.where(mask_array, tokenizer.convert_tokens_to_ids(tokenizer.mask_token), masked_sentence)
            masked_sentence = tokenizer.batch_decode(masked_sentence[:,1:-1])
            masked_sentence = [merge_mask_token(s) for s in masked_sentence]
            masked_sentences.append(masked_sentence)

        return masked_sentences

    def mask_by_entropy(example, p=0.5):
        """
        example: a string with multiple sentences. i.e. "sent1. sent2. blahblah. sent3. sent4."
        """
        # sent_tokenize
        sent_tokenized_example = sent_tokenize_and_strip(example) # list of string

        # mask each sent_tokenized sentence
        masked_candidates = mask_sentences_random(sent_tokenized_example, tokenizer, p=p) # list of list of string
        # append original sent for entropy filtering
        for i in range(len(sent_tokenized_example)):
            masked_candidates[i].insert(0, sent_tokenized_example[i])
        # get embedding
        embedded_candidates = [embed(cands) for cands in masked_candidates]
        # get entropy
        ent_scores = [entropy(cands) for cands in embedded_candidates]
        # filter by argmax entropy scores
        filtered_indeces = [torch.argmax(ent_s).item() for ent_s in ent_scores]

        # re-join
        masked_example = []
        for i, ind in enumerate(filtered_indeces):
            masked_example.append(masked_candidates[i][ind])
        masked_example = ' '.join(masked_example)

        return masked_example

    def preprocess_data(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_dataset = yelp_dataset.map(preprocess_data, batched=True)

    masked_outputs = []
    examples = tokenized_dataset['train']['text'][450000:]
    labels = tokenized_dataset['train']['label'][450000:]
    excepted_texts_train = []
    for i in tqdm(range(len(examples)), desc="masking"):
        example = examples[i]
        label = labels[i]
        try:
            masked_text = mask_by_entropy(example, p=0.7)
            masked_outputs.append({"text": example,
                                "label": label,
                                "masked_text": masked_text,
                                })
        except:
            excepted_texts_train.append({"text": example, "label": label})
            continue
    out_file = os.path.join(DATA_DIR, 'masked_yelp_train_7.json_3')
    with open(out_file, 'w') as json_f:
        json.dump(masked_outputs, json_f)
    print("done!")


if __name__ == "__main__":
    main()
