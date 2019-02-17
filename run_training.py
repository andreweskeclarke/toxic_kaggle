#!/usr/bin/env python3
import argparse
import collections
import google_research.bert.modeling as modeling
import google_research.bert.optimization as optimization
import json
import numpy as np
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


class Model:
    def __init__(self, bert_config, train_config, training_data):
        self.n_inputs = bert_config.max_position_embeddings
        self.batch_size = train_config['batch_size']
        self.n_output_labels = train_config['n_output_labels']
        self.learning_rate = train_config['learning_rate']
        self.n_training_steps = train_config['n_training_steps']
        self.n_warmup_steps = int(train_config['n_warmup_proportion'] * self.n_training_steps)

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
            self.probs = tf.math.sigmoid(self.logits)
            self.loss = tf.losses.sigmoid_cross_entropy(
                    self.labels,
                    self.logits)

    def train(self):
        with tf.Session() as sess:
            # global_step = tf.train.get_or_create_global_step()
            # optimizer = optimization.AdamWeightDecayOptimizer(
            #     learning_rate=self.learning_rate,
            #     weight_decay_rate=0.01,
            #     beta_1=0.9,
            #     beta_2=0.999,
            #     epsilon=1e-6,
            #     exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
            # tvars = tf.trainable_variables()
            # grads = tf.gradients(self.loss, tvars)

            # # This is how the model was pre-trained.
            # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            # train_op = optimizer.apply_gradients(
            #     zip(grads, tvars), global_step=global_step)

            # # Normally the global step update is done inside of `apply_gradients`.
            # # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
            # # a different optimizer, you should probably take this line out.
            # new_global_step = global_step + 1
            # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
            train_op = optimization.create_optimizer(
                    self.loss,
                    self.learning_rate, 
                    self.n_training_steps,
                    self.n_warmup_steps,
                    use_tpu=False)

            sess.run(tf.global_variables_initializer())
            print("Running training...")
            losses = deque(maxlen=500)
            labels = deque(maxlen=500)
            probs = deque(maxlen=500)
            n_steps = 0
            while True:
                _, l, p, t = sess.run([train_op, self.loss, self.probs, self.labels])
                for i in range(self.batch_size):
                    losses.append(l)
                    probs.append(p[i, :])
                    labels.append(t[i, :])
                n_steps += 1
                if n_steps % 10 == 0:
                    mean_xent = sum(losses) / len(losses)
                    auc = "n/a"
                    try:
                        auc = "{:5.7f}".format(sklearn.metrics.roc_auc_score(np.array(labels), np.array(probs)))
                    except ValueError as e:
                        print(e)
                    print('Trained step {:>10n}: loss {:>5.7f}, auc {:>12s}'.format(n_steps, mean_xent, auc))



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
