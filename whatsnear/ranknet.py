# coding=utf-8
import tensorflow as tf
import numpy as np
import sqlite3
from haversine import haversine
import geohash
import math
import random
import time
import os

feature_num = 10
h1_num = 10


class RankNet(object):
    def __init__(self):
        self._labels = []
        self._features = []

    def _read_train_data(self, train_file):
        print('[RankNet] Pre-calculated train file found, reading from external file...')
        start_time = time.clock()

        with open(train_file, 'r') as f:
            for line in f.readlines():
                parts = line.split(' ')
                self._labels.append([float(parts[0])])
                self._features.append([float(feature) for feature in parts[1:]])

        end_time = time.clock()
        print('[RankNet] Training data read in %f seconds.' % (end_time - start_time))

    def _write_train_data(self, train_file):
        with open(train_file, 'w') as f:
            for i in range(0, len(self._labels)):
                row = []
                row.extend(self._labels[i])
                row.extend(self._features[i])
                f.write(' '.join(row))
        print('[RankNet] Calculated training data stored into %s.' % train_file)

    def _calculate_train_data(self, database):
        from progress.bar import Bar

        print('[RankNet] Pre-calculated train file not found, calculating training data...')
        start_time = time.clock()

        conn = sqlite3.connect(database)
        c = conn.cursor()
        # if geohash has never been calculated
        if c.execute('''SELECT geohash FROM \'Beijing-Checkins\' LIMIT 1''').fetchone()[0] is None:
            # calculate the geohash value and store in database
            for row in c.execute('''SELECT id,lng,lat FROM \'Beijing-Checkins\''''):
                conn.execute('''UPDATE \'Beijing-Checkins\' set geohash=? WHERE id=?''',
                             (geohash.encode(float(row[2]), float(row[1])), row[0]))
            conn.commit()

        # radius to calculate features
        r = 200

        total_num = int(conn.execute('''SELECT COUNT(*) FROM \'Beijing-Checkins\'''').fetchone()[0])
        #bar = Bar('Calculating', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        for row in conn.execute('''SELECT lat,lng,category,checkins,geohash FROM \'Beijing-Checkins\''''):
            # add label
            self._labels.append([int(row[3])])

            # calculate features
            x = []

            # calculate global parameters


            # calculate neighboring points
            neighbors = []
            potential_neighbors = []

            for point in conn.execute('''SELECT lat,lng,category,checkins FROM \'Beijing-Checkins\' 
                                         WHERE geohash LIKE \'%s%%\'''' % row[4][:6]):
                potential_neighbors.append(point)

            for neighbor in potential_neighbors:
                if haversine((float(neighbor[0]), float(neighbor[1])), (float(row[0]), float(row[1]))) * 1000 <= r:
                    neighbors.append(neighbor)

            # density
            x.append(len(neighbors))

            # neighbors entropy
            entropy = 0
            categories = {}
            for neighbor in neighbors:
                if neighbor[2] not in categories:
                    categories[neighbor[2]] = 1
                else:
                    categories[neighbor[2]] += 1

            for (key, value) in categories.items():
                entropy += float(value) / len(neighbors) * -1 * math.log(float(value) / len(neighbors), 10)

            x.append(entropy)

            # competitiveness
            competitiveness = 0
            category = u'生活娱乐'
            if category in categories:
                competitiveness = float(categories[category]) / len(neighbors)

            x.append(competitiveness)

            # quality by jensen
            popularity = 0
            for neighbor in neighbors:
                popularity += int(neighbor[3])

            x.append(popularity)


            # area popularity




            self._features.append(x)
            print(x)
            #bar.next()

        #bar.finish()

        end_time = time.clock()
        print('[RankNet] Training data calculated in %f seconds.' % (end_time - start_time))

        # store calculated train data
        self._write_train_data(os.path.dirname(database) + '/train.txt')

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

    def train(self, database, train_file):
        if train_file is None or not os.path.exists(train_file):
            self._calculate_train_data(database)
        else:
            self._read_train_data(train_file)

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
