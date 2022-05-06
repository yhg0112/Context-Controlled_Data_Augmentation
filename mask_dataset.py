import argparse
import jsonlines

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DATA_CACHE_DIR = "./data/.cache"


softmax = torch.nn.Softmax(dim=-1)
def entropy(p):
    return torch.sum(-1*p*torch.log(p), dim=-1)

def sent_tokenize_and_strip(sentences):
    sentences = sentences.replace("\\n\\n", " ")
    return [s.strip() for s in sent_tokenize(sentences)]


def main(args):
    device = torch.device("cuda")

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(device)

    def embed(sentences):
        with torch.no_grad():
            tokenized = tokenizer(sentences, padding=True, max_length=args.max_len, return_tensors='pt').to(device)
            rslt = sentiment_model(tokenized['input_ids'][:,:args.max_len])
            return softmax(rslt['logits'])

    def mask_sentences_random(sentences, tokenizer, n_sentences, p):
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

    def mask_by_entropy(example, n_sentences, p):
        """
        example: a string with multiple sentences. i.e. "sent1. sent2. blahblah. sent3. sent4."
        """
        # sent_tokenize
        sent_tokenized_example = sent_tokenize_and_strip(example) # list of string

        # mask each sent_tokenized sentence
        masked_candidates = mask_sentences_random(sent_tokenized_example, tokenizer, n_sentences=n_sentences, p=p) # list of list of string
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

    yelp_dataset = load_dataset('json', data_files=args.in_data, cache_dir=DATA_CACHE_DIR)
    tokenized_dataset = yelp_dataset.map(preprocess_data, batched=True)

    masked_outputs = []
    excepted_texts_train = []
    for d in tqdm(tokenized_dataset['train'], desc="masking"):
        try:
            masked_text = mask_by_entropy(d['text'], args.n_sentences, args.mask_p)
            masked_outputs.append({
                "text": d['text'],
                "label": d['label'],
                "masked_text": masked_text,
            })
        except:
            excepted_texts_train.append(d)
    print(f"{len(excepted_texts_train):,} out of {len(tokenized_dataset):,} samples skipped due to error!")
    with jsonlines.open(args.out_data, 'w') as writer:
        writer.write_all(masked_outputs)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    parser.add_argument('--mask_p', type=float, required=True)
    parser.add_argument('--n_sentences', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=512)
    args = parser.parse_args()
    main(args)
