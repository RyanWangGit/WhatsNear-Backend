import random
import time
import numpy as np


class RankNet(object):
    def __init__(self):
        # training data
        self._dataset = None
        self._is_ready = False

        # model
        self._model = None

        # score function
        self._score_function = None

    def _train_model(self, features, labels, epochs=10, batches=10):
        # Michael A. Alcorn (malcorn@redhat.com)
        # A (slightly modified) implementation of RankNet as described in [1].
        #   [1] http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
        #   [2] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        from tensorflow.python.keras import backend
        from tensorflow.python.keras.layers import Activation, Add, Dense, Input, Lambda
        from tensorflow.python.keras.models import Model

        dimension = len(features[0])

        # model
        h_1 = Dense(128, activation="relu")
        h_2 = Dense(64, activation="relu")
        h_3 = Dense(32, activation="relu")
        s = Dense(1)

        # relevant document score
        rel_doc = Input(shape=(dimension,), dtype="float32")
        h_1_rel = h_1(rel_doc)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # irrelevant document score
        irr_doc = Input(shape=(dimension,), dtype="float32")
        h_1_irr = h_1(irr_doc)
        h_2_irr = h_2(h_1_irr)
        h_3_irr = h_3(h_2_irr)
        irr_score = s(h_3_irr)

        # subtract scores
        negated_irr_score = Lambda(lambda x: -1 * x, output_shape=(1,))(irr_score)
        diff = Add()([rel_score, negated_irr_score])

        # pass difference through sigmoid function
        prob = Activation("sigmoid")(diff)

        # build model.
        model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
        model.compile(optimizer="adadelta", loss="binary_crossentropy")

        # feed in data
        x1 = []
        x2 = []
        y = []
        for i in range(len(features) - 1):
            x1.append(features[i])
            x2.append(features[i + 1])
            if labels[i][0] == 0 and labels[i][0] == labels[i + 1][0]:
                y.append([0.5])
            else:
                y.append([float(labels[i][0]) / (labels[i][0] + labels[i + 1][0])])

        X1 = np.matrix(x1)
        X2 = np.matrix(x2)
        y = np.matrix(y)

        history = model.fit([X1, X2], y, batch_size=batches, epochs=epochs, verbose=1)
        # train

        self._model = model

        # generate scores from document/query features
        self._score_function = backend.function([rel_doc], [rel_score])

    def load(self, path):
        from tensorflow.python.keras.models import load_model
        print('[TensorFlow] Trained model file found, loading model...')
        self._model = load_model(path)
        self._is_ready = True
        print('[TensorFlow] Trained model loaded.')

    def save(self, path):
        print('[TensorFlow] Saving model...')
        self._model.save(path)
        print('[TensorFlow] Model saved to %s' % path)

    def ndcg(self, y_true, y_score, k=10):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        if ideal_dcg == 0:
            ndcg = 1
        else:
            ndcg = float(dcg) / float(ideal_dcg)
        return ndcg

    def train(self, dataset, rate=1, epochs=3, batches=10):
        print('[TensorFlow] Start training model...')
        start_time = time.clock()
        self._dataset = dataset
        train_features = dataset.get_features()[:int(len(dataset.get_features()) * rate)]
        train_labels = dataset.get_labels()[:int(len(dataset.get_features()) * rate)]
        test_features = dataset.get_features()[int(len(dataset.get_features()) * rate):]
        test_labels = dataset.get_labels()[int(len(dataset.get_features()) * rate):]

        self._train_model(train_features, train_labels, epochs=epochs, batches=batches)
        self._is_ready = True
        end_time = time.clock()
        print('[TensorFlow] Model trained in %f seconds' % (end_time - start_time))

        print('[TensorFlow] Start testing...')
        test_range = range(len(test_features))

        ndcg = 0
        for _ in range(1000):
            test = random.sample(test_range, 10)
            to_rank_features = []
            to_rank_labels = []
            for index in test:
                to_rank_features.append(test_features[index])
                to_rank_labels.append(test_labels[index])

            scores = self._score_function([to_rank_features])[0]
            ndcg += self.ndcg(np.array(to_rank_labels),  np.array(scores))

        print('[TensorFlow] testing ended with NDCG %f' % float(ndcg / 1000))
        return float(ndcg / 1000)

    def rank(self, query_points, caller):
        if not self._is_ready:
            print('[TensorFlow - 0x%x] Ranker isn\'t ready, train the model or load the pre-trained model first.' % id(caller))
            return None

        print('[TensorFlow - 0x%x] Start ranking the query points with size %d.' % (id(caller), len(query_points)))

        features = []
        for point in query_points:
            x = self._dataset.vectorize_point(point['neighbors'], u'生活娱乐')
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
