#!/usr/bin/env python3
import argparse
import collections
import datetime
import json
import names
import os
import shutil
from toxic.model import Model


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
    parser.add_argument('--validation_data', type=str, default='data/validation.tf_record',
            help='Location of the validation data')
    args = parser.parse_args()
    experiment_dir, train_config_file = setup_experiment(args.train_config_file)
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    train_config = json.load(open(train_config_file))
    model = Model(bert_config, train_config, args.training_data, args.validation_data, experiment_dir)
    model.train()


if __name__ == '__main__':
    main()
