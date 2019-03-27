import numpy as np
import sklearn
import tensorflow as tf


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
