# coding=utf-8
import tensorflow as tf
import numpy as np
import random
import time

feature_num = 10
h1_num = 10


class RankNet(object):
    def __init__(self):
        self._points = []

    def _prepare_train_data(self, points):
        start_time = time.clock()
        print('[TensorFlow] Preparing training data...')

        end_time = time.clock()
        print('[TensorFlow] Traning data prepared in %f seconds.' % (end_time - start_time))
        pass

    def get_train_data(self, batch_size=32):
        # generate data with 10 dimensions
        X1, X2 = [], []
        Y1, Y2 = [], []

        for i in range(0, batch_size):
            x1 = []
            x2 = []
            o1 = 0.0
            o2 = 0.0
            for j in range(0, 10):
                r1 = random.random()
                r2 = random.random()
                x1.append(r1)
                x2.append(r2)

                mu = 2.0
                if j >= 5: mu = 3.0
                o1 += r1 * mu
                o2 += r2 * mu
            X1.append(x1)
            Y1.append([o1])
            X2.append(x2)
            Y2.append([o2])

        return (np.array(X1), np.array(X2)), (np.array(Y1), np.array(Y2))

        self._prepare_train_data(points)
    def train(self, database, train_file):

        print('[TensorFlow] Start training model...')
        start_time = time.clock()

        with tf.name_scope("input"):
            x1 = tf.placeholder(tf.float32, [None, feature_num], name="x1")
            x2 = tf.placeholder(tf.float32, [None, feature_num], name="x2")

            o1 = tf.placeholder(tf.float32, [None, 1], name="o1")
            o2 = tf.placeholder(tf.float32, [None, 1], name="o2")

        # add layer1
        with tf.name_scope("layer1"):
            with tf.name_scope("w1"):
                w1 = tf.Variable(tf.random_normal([feature_num, h1_num]), name="w1")
                # tf.summary.histogram("layer1/w1", w1)
            with tf.name_scope("b1"):
                b1 = tf.Variable(tf.random_normal([h1_num]), name="b1")
                # tf.summary.histogram("layer1/b1", b1)

            # didn't add activation function
            with tf.name_scope("h1_o1"):
                h1_o1 = tf.matmul(x1, w1) + b1
                # tf.summary.histogram("h1_o1", h1_o1)

            with tf.name_scope("h2_o1"):
                h1_o2 = tf.matmul(x2, w1) + b1
                # tf.summary.histogram("h2_o1", h1_o2)

        # add output layer
        with tf.name_scope("output"):
            with tf.name_scope("w2"):
                w2 = tf.Variable(tf.random_normal([h1_num, 1]), name="w2")
                # tf.summary.histogram("output/w2", w2)

            with tf.name_scope("b2"):
                b2 = tf.Variable(tf.random_normal([1]))
                # tf.summary.histogram("output/b2", b2)

            h2_o1 = tf.matmul(h1_o1, w2) + b2
            h2_o2 = tf.matmul(h1_o2, w2) + b2

        # calculate probability based on output layer
        with tf.name_scope("loss"):
            o12 = o1 - o2
            h_o12 = h2_o1 - h2_o2

            pred = 1 / (1 + tf.exp(-h_o12))
            lable_p = 1 / (1 + tf.exp(-o12))

            cross_entropy = -lable_p * tf.log(pred) - (1 - lable_p) * tf.log(1 - pred)
            reduce_sum = tf.reduce_sum(cross_entropy, 1)
            loss = tf.reduce_mean(reduce_sum)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("train_op"):
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        with tf.Session() as sess:
            # summary_op = tf.summary.merge_all()
            # writer = tf.summary.FileWriter("./logs/", sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(0, 100):
                X, Y = self.get_train_data()
                sess.run(train_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
                if epoch % 10 == 0:
                    # summary_result = sess.run(summary_op, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
                    # writer.add_summary(summary_result, epoch)
                    l_v = sess.run(loss, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
                    h_o12_v = sess.run(h_o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
                    o12_v = sess.run(o12, feed_dict={x1: X[0], x2: X[1], o1: Y[0], o2: Y[1]})
                    #print "------ epoch[%d] loss_v[%f] ------ " % (epoch, l_v)
                    #for k in range(0, len(o12_v)):
                        #print "k[%d] o12_v[%f] h_o12_v[%f]" % (k, o12_v[k], h_o12_v[k])

        end_time = time.clock()

        print '[TensorFlow] Model trained in %f seconds' % (end_time - start_time)
        return

    def rank(self, query_points, caller):
        print('[TensorFlow - 0x%x] Start ranking the query points with size %d.' % (id(caller), len(query_points)))

        print('[TensorFlow - 0x%x] Ranking finished.' % id(caller))

        return query_points
