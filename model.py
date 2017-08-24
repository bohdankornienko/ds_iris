import tensorflow as tf
import numpy as np

import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(1)


class TensorFlowModel:
    def __init__(self, input_size, learning_rate, classes_count, n_epoch, batch_size, skip_step):
        self._input_size = input_size
        self._batch_size = batch_size
        self._skip_step = skip_step
        self._lr = learning_rate
        self._classes_count = classes_count
        self._n_epoch = n_epoch
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self._config = tf.ConfigProto()
        self._config.log_device_placement = False
        self._config.gpu_options.allow_growth = True

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self._X = tf.placeholder(tf.float32, [None, self._input_size], name="input")
            self._Y = tf.placeholder(tf.float32, [None, self._classes_count], name="output")

    def _create_model(self):
        layer_1_size = 512

        with tf.variable_scope('layer_1'):
            w = tf.get_variable(name='weights', shape=[self._input_size, layer_1_size],
                                initializer=tf.truncated_normal_initializer())

            b = tf.get_variable(name='biases', shape=[layer_1_size],
                                initializer=tf.random_normal_initializer())

            fc = tf.nn.relu(tf.matmul(self._X, w) + b)

        with tf.variable_scope('softmax_linear'):
            w = tf.get_variable('weights', [layer_1_size, self._classes_count],
                                initializer=tf.truncated_normal_initializer())

            b = tf.get_variable('biases', [self._classes_count],
                                initializer=tf.random_normal_initializer())

            self._logits = tf.matmul(fc, w) + b

    def _create_loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._Y, logits=self._logits)
            self._loss = tf.reduce_mean(entropy, name='loss')

    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            self._optimizer = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss,
                                                                                   global_step=self._global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self._loss)
            tf.summary.histogram("histogram loss", self._loss)
            self._summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train_model(self, data_set):
        with tf.Session(config=self._config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('./iris', sess.graph)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('iris/checkpoint'))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self._global_step.eval()

            start_time = time.time()

            total_loss = 0.0

            epoch_no = 0
            index = initial_step
            while epoch_no < self._n_epoch:
                x_batch, y_batch = data_set.next_batch(self._batch_size)

                # y_batch to one hot vector
                res = np.zeros((y_batch.shape[0], 3))
                res[np.arange(res.shape[0]), y_batch.ravel()] = 1

                y_batch = res

                _, loss_batch, summary = sess.run([self._optimizer, self._loss, self._summary_op],
                                                  feed_dict={self._X: x_batch, self._Y: y_batch})
                writer.add_summary(summary, global_step=index)
                total_loss += loss_batch
                if (index + 1) % self._skip_step == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / self._skip_step))
                    total_loss = 0.0
                    saver.save(sess, 'iris/iris', index)

                if data_set.is_end_of_data:
                    epoch_no += 1
                index += 1

            print("Optimization Finished!")
            print("Total time: {0} seconds".format(time.time() - start_time))

    def test_model(self, data_set):
        examples_num = 0
        total_correct_preds = 0

        with tf.Session(config=self._config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('iris/checkpoint'))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                 saver.restore(sess, ckpt.model_checkpoint_path)

            while not data_set.is_end_of_data:
                features, labels = data_set.next_batch()

                # y_batch to one hot vector
                res = np.zeros((labels.shape[0], 3))
                res[np.arange(res.shape[0]), labels.ravel()] = 1

                labels = res

                examples_num += features.shape[0]

                _, _, logits_batch = sess.run([self._optimizer, self._loss, self._logits],
                                              feed_dict={self._X: features, self._Y: labels})

                preds = tf.nn.softmax(logits_batch)
                correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                total_correct_preds += sess.run(accuracy)

        print("Accuracy {} for {} exaples".format(total_correct_preds / examples_num, examples_num))
