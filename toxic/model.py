import google_research.bert.modeling as modeling
import google_research.bert.optimization as optimization
import numpy as np
import sklearn
import tensorflow as tf
from collections import deque
from toxic.summary_writer import SummaryWriter


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
    def __init__(self, bert_config, train_config, training_data, validation_data, experiment_dir):
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
