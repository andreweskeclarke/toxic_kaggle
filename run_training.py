#!/usr/bin/env python3
import argparse
import collections
import datetime
import google_research.bert.modeling as modeling
import google_research.bert.optimization as optimization
import json
import names
import numpy as np
import os
import shutil
import sklearn
import tensorflow as tf
from collections import deque

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


class SummaryWriter():
    def __init__(self, session, experiment_dir, n_output_labels):
        self.session = session
        self.n_output_labels = n_output_labels
        self.writer = tf.summary.FileWriter(os.path.join(experiment_dir, 'tflogs'), self.session.graph)
        self.mean_xent_tf = tf.placeholder(tf.float32)
        self.mean_xent_summary = tf.summary.scalar('mean_xent', self.mean_xent_tf)
        self.auc_tf = tf.placeholder(tf.float32)
        self.auc_summary = tf.summary.scalar('auc_roc', self.auc_tf)
        self.auc_summaries_tf = list()
        self.auc_summaries = list()
        for i in range(self.n_output_labels):
            p = tf.placeholder(tf.float32)
            self.auc_summaries_tf.append(p)
            self.auc_summaries.append(tf.summary.scalar('aucs_roc/{}'.format(i), p))
        self.merged_summary = tf.summary.merge_all()

    def add_summary(self, step, losses, probs, labels):
        mean_xent = sum(losses) / len(losses)
        auc = 0
        aucs = [0] * self.n_output_labels
        try:
            auc = sklearn.metrics.roc_auc_score(np.array(labels), np.array(probs))
        except: pass
        for i in range(self.n_output_labels):
            try:
                aucs[i] = sklearn.metrics.roc_auc_score(np.array(labels)[:, i], np.array(probs)[:, i])
            except: pass
        feed_dict = dict()
        feed_dict[self.mean_xent_tf] = mean_xent
        feed_dict[self.auc_tf] = auc
        for i, a in enumerate(self.auc_summaries_tf):
            feed_dict[a] = aucs[i]
        current_summaries = self.session.run(self.merged_summary, feed_dict=feed_dict)
        self.writer.add_summary(current_summaries, step)
        if step % 10 == 0:
            auc_s = '{:1.5f}'.format(auc)
            aucs_s = ','.join(['{:1.3f}'.format(x) for x in aucs])
            print('Trained step {:>9n}: loss {:>2.6f}, auc {:>12s}, ({})'.format(step, mean_xent, auc_s, aucs_s))


class Model:
    def __init__(self, bert_config, train_config, training_data, experiment_dir):
        self.n_inputs = bert_config.max_position_embeddings
        self.batch_size = train_config['batch_size']
        self.n_output_labels = train_config['n_output_labels']
        self.learning_rate = train_config['learning_rate']
        self.n_train_obs_for_plotting = train_config['n_train_obs_for_plotting']
        self.n_training_steps = int(train_config['n_examples_per_epoch'] * train_config['n_epochs'] / self.batch_size)
        self.n_warmup_steps = int(train_config['n_warmup_proportion'] * train_config['n_examples_per_epoch'])
        self.experiment_dir = experiment_dir

        with tf.variable_scope('input_parsing'):
            dataset = tf.data.TFRecordDataset([training_data])
            dataset = dataset.map(build_decoder(self.n_inputs, self.n_output_labels))
            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            input_ids = next_element['input_ids']
            masks = next_element['input_mask']
            segment_ids = next_element['segment_ids']
            self.batch_labels_tf = next_element['label_ids']

        self.bert_model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=input_ids,
                input_mask=masks,
                token_type_ids=segment_ids,
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
            self.batch_probs_tf = tf.math.sigmoid(self.logits)
            self.batch_loss_tf = tf.losses.sigmoid_cross_entropy(
                    self.batch_labels_tf,
                    self.logits)

    def train(self):
        print('Running training...')
        train_losses = deque(maxlen=int(32/self.batch_size))
        train_labels = deque(maxlen=int(self.n_train_obs_for_plotting / self.batch_size))
        train_probs = deque(maxlen=int(self.n_train_obs_for_plotting / self.batch_size))
        with tf.Session() as session:
            train_op = optimization.create_optimizer(self.batch_loss_tf, self.learning_rate, self.n_training_steps, self.n_warmup_steps, use_tpu=False)
            writer = SummaryWriter(session, self.experiment_dir, self.n_output_labels)
            session.run(tf.global_variables_initializer())
            for step in range(self.n_training_steps):
                _, l, p, t = session.run([train_op, self.batch_loss_tf, self.batch_probs_tf, self.batch_labels_tf])
                for i in range(self.batch_size):
                    train_losses.append(l)
                    train_probs.append(p[i, :])
                    train_labels.append(t[i, :])
                writer.add_summary(step, train_losses, train_probs, train_labels)

def setup_experiment(training_config_file):
    experiment = names.get_full_name().lower().replace(' ', '_') + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_dir = os.path.join(os.getcwd(), 'training', experiment)
    os.mkdir(experiment_dir)
    new_training_config_file = os.path.join(experiment_dir, os.path.basename(training_config_file))
    shutil.copyfile(training_config_file, new_training_config_file)
    return experiment_dir, new_training_config_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_config_file', type=str, default='models/bert_config.json',
            help='Location of the bert config file')
    parser.add_argument('--train_config_file', type=str, default='train_config.json',
            help='Location of the training config file')
    parser.add_argument('--training_data', type=str, default='data/train.tf_record',
            help='Location of the training data')
    args = parser.parse_args()
    experiment_dir, train_config_file = setup_experiment(args.train_config_file)
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    train_config = json.load(open(train_config_file))
    model = Model(bert_config, train_config, args.training_data, experiment_dir)
    model.train()


if __name__ == '__main__':
    main()
