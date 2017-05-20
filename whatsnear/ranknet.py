# coding=utf-8
import tensorflow as tf
import random
import time


class RankNet(object):
    def __init__(self, dataset):
        # training data
        self._is_ready = False
        self._dataset = dataset

        # model
        self._model = None

        # score function
        self._score_function = None

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

    def load(self, path):
        from keras.models import load_model
        print('[TensorFlow] Trained model file found, loading model...')
        self._model = load_model(path)
        self._is_ready = True
        print('[TensorFlow] Trained model loaded.')

    def save(self, path):
        print('[TensorFlow] Saving model...')
        self._model.save(path)
        print('[TensorFlow] Model saved to %s' % path)

    def train(self, dataset, epochs=1000, batches=10):
        print('[TensorFlow] Start training model...')
        start_time = time.clock()
        self._keras_train_model(dataset.get_features(), dataset.get_labels(), epochs=epochs, batches=batches)
        self._is_ready = True
        end_time = time.clock()
        print '[TensorFlow] Model trained in %f seconds' % (end_time - start_time)

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
