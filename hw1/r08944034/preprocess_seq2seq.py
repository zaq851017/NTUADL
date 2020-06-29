import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
from tqdm import tqdm
from argparse import ArgumentParser


def main(args):

    # loading datasets from jsonl files
    with open(args.input_data_path) as f:
        valid = [json.loads(valid) for valid in f]

    logging.info('Collecting documents...')
    documents = (
        [sample['text'] for sample in valid]
    )

    logging.info('Collecting words in documents...')
    tokenizer = Tokenizer(lower=True)
    words = tokenizer.collect_words(documents)

    logging.info('Loading embedding...')
    """
    embedding = Embedding("./glove.6B.300d.txt", words=words)
    with open('./embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)
    """
    with open('./embedding.pkl', 'rb') as file:
        embedding=pickle.load(file)
    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating valid dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, valid),
        'data.pkl',
        tokenizer.pad_token_id
    )


def process_samples(tokenizer, samples):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    processeds = []
    for sample in tqdm(samples):
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']) + [eos_id],
        }
        if 'summary' in sample:
            processed['summary'] = (
                [bos_id]
                + tokenizer.encode(sample['summary'])
                + [eos_id]
            )
        processeds.append(processed)

    return processeds


def create_seq2seq_dataset(samples, save_path,padding=0):
    dataset = Seq2SeqDataset(
        samples, padding=padding,
        max_text_len=300,
        max_summary_len=80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('output_dir', type=Path,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_data_path')
    args  = parser.parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
