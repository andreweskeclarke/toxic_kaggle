#!/usr/bin/env python3
import argparse
import collections
import csv
import tensorflow as tf
from google_research.bert.tokenizer import FullTokenizer

def main():
    MAX_SEQUENCE_LENGTH = 512    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--input_file', help='A file containing many lines for tokenization',
                        type=str)
    parser.add_argument('--output_file', help='The output TF Record file',
                        type=str)
    parser.add_argument('--vocab_file', help='A file containing the dictionary for tokenization',
                        type=str, default='models/vocab.txt')
    args = parser.parse_args()
    tokenizer = FullTokenizer(args.vocab_file, args.do_lower_case)
    writer = tf.python_io.TFRecordWriter(args.output_file)
    with open(args.input_file) as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0: continue
            tokens = []
            tokens.append("[CLS]")
            tokens.extend(tokenizer.tokenize(row[1])[0:(MAX_SEQUENCE_LENGTH-2)])
            tokens.append("[SEP]")
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < MAX_SEQUENCE_LENGTH:
                input_ids.append(0)
                mask.append(0)
                segment_ids.append(0)
            targets = list([int(i) for i in row[2:]])

            features = collections.OrderedDict()
            features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
            features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(mask)))
            features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))
            features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(targets)))
            features["is_real_example"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    main()
