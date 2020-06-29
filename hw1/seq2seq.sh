#!/usr/bin/env bash
python3.7 preprocess_seq2seq.py --input_data_path $1
python3.7 model2_test.py --output_path $2