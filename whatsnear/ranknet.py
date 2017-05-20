# coding=utf-8
import tensorflow as tf
from database import Database
import json
import math
import random
import time
import os


class RankNet(object):
    def __init__(self):
        # training data
        self._labels = []
        self._features = []
        self._is_ready = False
        self._database = None

        # global parameters
        self._mean_category_number = {}
        self._category_coefficient = {}
        self._categories = {}
        self._all_points = []

        # model
        self._model = None

        # score function
        self._score_function = None

    def _read_train_data(self, train_file):
        print('[RankNet] Pre-calculated train file found, reading from external file...')
        start_time = time.clock()

        with open(train_file, 'r') as f:
            self._mean_category_number = json.loads(f.readline())
            self._category_coefficient = json.loads(f.readline())
            self._categories = json.loads(f.readline())
            self._labels = json.loads(f.readline())
            self._features = json.loads(f.readline())

        end_time = time.clock()
        print('[RankNet] Training data read in %f seconds.' % (end_time - start_time))

    def _write_train_data(self, train_file):
        with open(train_file, 'w') as f:
            f.write(json.dumps(self._mean_category_number) + '\n')
            f.write(json.dumps(self._category_coefficient) + '\n')
            f.write(json.dumps(self._categories) + '\n')
            f.write(json.dumps(self._labels) + '\n')
            f.write(json.dumps(self._features))
        print('[RankNet] Calculated training data stored into %s.' % train_file)

    def _neighbor_categories(self, neighbors):
        # calculate sub-global parameters
        neighbor_categories = {}
        for category, _ in self._categories.items():
            neighbor_categories[category] = 0
        for neighbor in neighbors:
            neighbor_categories[neighbor['category']] += 1
        return neighbor_categories

    def _vectorize_point(self, neighbors, training_category):
        neighbor_categories = self._neighbor_categories(neighbors)

        x = []

        # density
        x.append(len(neighbors))

        # neighbors entropy
        entropy = 0
        for (key, value) in neighbor_categories.items():
            if value == 0:
                continue
            entropy += (float(value) / len(neighbors)) * -1 * math.log(float(value) / len(neighbors))

        x.append(entropy)

        # competitiveness
        competitiveness = 0
        if training_category in neighbor_categories:
            competitiveness = -1 * float(neighbor_categories[training_category]) / len(neighbors)

        x.append(competitiveness)

        # quality by jensen
        jenson_quality = 0
        for category, _ in self._categories.items():
            if self._category_coefficient[category][training_category] == 0:
                continue

            jenson_quality += math.log(self._category_coefficient[category][training_category]) * (
            neighbor_categories[category] - self._mean_category_number[category][training_category])

        x.append(jenson_quality)

        # area popularity
        popularity = 0
        for neighbor in neighbors:
            popularity += int(neighbor['checkins'])

        x.append(popularity)
        return x

    def _calculate_train_data(self):
        from progress.bar import Bar

        print('[RankNet] Pre-calculated train file not found, calculating training data...')
        start_time = time.clock()

        self._database.update_geohash()

        # radius to calculate features
        r = 200

        # calculate global parameters
        total_num = self._database.get_total_num()
        #total_num = 10000
        self._categories = self._database.get_categories()

        # calculate and store the neighboring points

        #bar = Bar('Calculating neighbors', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        #for row in self._database.get_connection().execute('''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
        #    self._all_points.append({
        #        'id': int(row[5]),
        #        'checkins': int(row[4]),
        #        'category': unicode(row[3]),
        #        'neighbors': self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
        #    })
        #    bar.next()
        #bar.finish()


        # calculate global category parameters
        for outer, _ in self._categories.items():
            self._mean_category_number[outer] = {}
            self._category_coefficient[outer] = {}
            for inner, _ in self._categories.items():
                self._mean_category_number[outer][inner] = 0
                self._category_coefficient[outer][inner] = 0

        # calculate mean category numbers
        bar = Bar('Calculating mean category numbers', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        for row in self._database.get_connection().execute(
                '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
            neighbors = self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
            for neighbor in neighbors:
                self._mean_category_number[neighbor['category']][unicode(row[3])] += 1
            bar.next()

        for p, _ in self._categories.items():
            for l, _ in self._categories.items():
                # TODO: to delete this line of code when we run training in full dataset
                if self._categories[l] == 0:
                    continue
                self._mean_category_number[p][l] /= self._categories[l]

        bar.finish()

        # calculate category coefficients
        bar = Bar('Calculating category coefficients', suffix='%(index)d / %(max)d, %(percent)d%%', max=len(self._categories) * len(self._categories))
        for p, _ in self._categories.items():
            for l, _ in self._categories.items():
                # TODO: delete this line of code in full dataset
                if self._categories[p] * self._categories[l] == 0:
                    continue

                k_prefix = float(total_num - self._categories[p]) / (self._categories[p] * self._categories[l])

                k_suffix = 0
                for row in self._database.get_connection().execute(
                        '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
                    if unicode(row[3]) == p:
                        neighbors = self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
                        neighbor_categories = self._neighbor_categories(neighbors)

                        if len(neighbors) - neighbor_categories[p] == 0:
                            continue

                        k_suffix += float(neighbor_categories[l]) / (len(neighbors) - neighbor_categories[p])

                self._category_coefficient[p][l] = k_prefix * k_suffix

                bar.next()

        bar.finish()

        bar = Bar('Calculating features', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        # calculate features
        for row in self._database.get_connection().execute(
                '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
            neighbors = self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
            # add label
            self._labels.append([int(row[4])])
            # add feature
            self._features.append(self._vectorize_point(neighbors, u'生活娱乐'))

            bar.next()

        bar.finish()

        end_time = time.clock()
        print('[RankNet] Training data calculated in %f seconds.' % (end_time - start_time))

        # store calculated train data
        self._write_train_data(os.path.dirname(self._database.get_file_path()) + '/train.txt')

    def get_train_data(self, batch_size=32):
        import numpy as np

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

    def _tf_train_model(self):

        feature_num = 10
        h1_num = 10

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
                    # print "------ epoch[%d] loss_v[%f] ------ " % (epoch, l_v)
                    # for k in range(0, len(o12_v)):
                    # print "k[%d] o12_v[%f] h_o12_v[%f]" % (k, o12_v[k], h_o12_v[k])

    def _keras_train_model(self, features, labels, epochs=100, batches=10):
        # Michael A. Alcorn (malcorn@redhat.com)
        # A (slightly modified) implementation of RankNet as described in [1].
        #   [1] http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
        #   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        import numpy as np

        from keras import backend
        from keras.layers import Activation, Add, Dense, Input, Lambda
        from keras.models import Model

        dimension = len(features[0])

        # Model.
        h_1 = Dense(128, activation="relu")
        h_2 = Dense(64, activation="relu")
        h_3 = Dense(32, activation="relu")
        s = Dense(1)

        # Relevant document score.
        rel_doc = Input(shape=(dimension,), dtype="float32")
        h_1_rel = h_1(rel_doc)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # Irrelevant document score.
        irr_doc = Input(shape=(dimension,), dtype="float32")
        h_1_irr = h_1(irr_doc)
        h_2_irr = h_2(h_1_irr)
        h_3_irr = h_3(h_2_irr)
        irr_score = s(h_3_irr)

        # Subtract scores.
        negated_irr_score = Lambda(lambda x: -1 * x, output_shape=(1,))(irr_score)
        diff = Add()([rel_score, negated_irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
        model.compile(optimizer="adadelta", loss="binary_crossentropy")

        # Fake data.
        x1 = []
        x2 = []
        indexes = xrange(len(features))
        for i in xrange(len(features) * 10):
            index1, index2 = random.sample(indexes, 2)

            if labels[index1] > labels[index2]:
                x1.append(features[index1])
                x2.append(features[index2])
            else:
                x1.append(features[index2])
                x2.append(features[index1])

        X1 = np.matrix(x1)
        X2 = np.matrix(x2)
        y = np.ones((X1.shape[0], 1))

        # Train model.
        history = model.fit([X1, X2], y, batch_size=batches, epochs=epochs, verbose=1)

        self._model = model

        # Generate scores from document/query features.
        self._score_function = backend.function([rel_doc], [rel_score])

    def _load_model(self, file_path):
        from keras.models import load_model
        self._model = load_model(file_path)

    def _save_model(self):
        self._model.save(os.path.dirname(self._database.get_file_path()) + 'model.h5')

    def train(self, database, train_file, model_file, epochs=1000, batches=10):
        # open database connection
        self._database = Database(database)

        if train_file is None or not os.path.exists(train_file):
            self._calculate_train_data()
        else:
            self._read_train_data(train_file)

        if model_file is not None:
            print('[TensorFlow] Trained model file found, loading model...')
            self._load_model(model_file)
            print('[TensorFlow] Trained model loaded.')
        else:
            print('[TensorFlow] Start training model...')
            start_time = time.clock()
            self._keras_train_model(self._features, self._labels, epochs=epochs, batches=batches)
            end_time = time.clock()
            print '[TensorFlow] Model trained in %f seconds' % (end_time - start_time)

        self._is_ready = True

        return

    def rank(self, query_points, caller):
        if not self._is_ready:
            print('[TensorFlow - 0x%x] Ranker isn\'t ready, train the model or load the pre-trained model first.' % id(caller))
            return None

        print('[TensorFlow - 0x%x] Start ranking the query points with size %d.' % (id(caller), len(query_points)))

        features = []
        for point in query_points:
            x = self._vectorize_point(point['neighbors'], u'生活娱乐')
            features.append(x)

            point['density'] = x[0]
            point['entropy'] = x[1]
            point['competitiveness'] = x[2]
            point['jenson'] = x[3]
            point['popularity'] = x[4]

        labels = self._score_function([features])[0]

        for i in range(len(labels)):
            query_points[i]['score'] = float(labels[i][0])

        query_points.sort(key=lambda p: p['score'], reverse=True)

        print('[TensorFlow - 0x%x] Ranking finished.' % id(caller))

        return query_points
