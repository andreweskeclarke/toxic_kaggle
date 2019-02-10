#!/usr/bin/env python3
import argparse
import collections
import json
import google_research.bert.modeling as modeling
import tensorflow as tf


def build_decoder(n_inputs, n_output_labels):
    def decoder(example):
        read_features = collections.OrderedDict()
        read_features['input_ids'] = tf.io.FixedLenFeature([n_inputs], dtype=tf.int64)
        read_features['input_mask'] = tf.io.FixedLenFeature([n_inputs], dtype=tf.int64)
        read_features['segment_ids'] = tf.io.FixedLenFeature([n_inputs], dtype=tf.int64)
        read_features['label_ids'] = tf.io.FixedLenFeature([n_output_labels], dtype=tf.int64)
        read_features['is_real_example'] = tf.io.FixedLenFeature([1], dtype=tf.int64)
        read_data = tf.parse_single_example(serialized=example, features=read_features)
        return read_data
    return decoder


class Model:
    def __init__(self, bert_config, train_config, training_data):
        self.n_inputs = bert_config.max_position_embeddings
        self.batch_size = train_config['batch_size']
        self.n_output_labels = train_config['n_output_labels']

        with tf.variable_scope('input_parsing'):
            dataset = tf.data.TFRecordDataset([training_data])
            dataset = dataset.map(build_decoder(self.n_inputs, self.n_output_labels))
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            self.next_element = iterator.get_next()
            self.input_ids = self.next_element['input_ids']
            self.masks = self.next_element['input_mask']
            self.segment_ids = self.next_element['segment_ids']
            self.labels = self.next_element['label_ids']

        self.bert_model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=self.input_ids,
                input_mask=self.masks,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=train_config['use_one_hot_embeddings'])

        with tf.variable_scope('bert_extension'):
            # bert_model.get_sequence_output() is a tensor of shape [batch_size, sequence_length, hidden_size]
            # bert_model.get_pooled_output() is a tensor of shape [batch_size, hidden_size], which
            #            is just the output corresponding to the first token, aka not really a pool
            self.bert_output = self.bert_model.get_pooled_output()
            self.logits = tf.layers.dense(
                    self.bert_output,
                    self.n_output_labels,
                    name='final_linear_layer')
            self.loss = tf.losses.sigmoid_cross_entropy(
                    self.labels,
                    self.logits)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    print(sess.run(self.bert_output))
                    print(sess.run(self.logits))
                    print(sess.run(self.loss))
                    input('Please press enter')
            except:
                pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_config_file', type=str, default='models/bert_config.json',
            help='Location of the bert config file')
    parser.add_argument('--train_config_file', type=str, default='training/train_config.json',
            help='Location of the training config file')
    parser.add_argument('--training_data', type=str, default='data/train.tf_record',
            help='Location of the training data')
    args = parser.parse_args()
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    train_config = json.load(open(args.train_config_file))
    model = Model(bert_config, train_config, args.training_data)
    model.train()


if __name__ == '__main__':
    main()
