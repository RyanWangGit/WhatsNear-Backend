import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class RankNet:
    def __init__(self):
        # training data
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

        dimension = features.shape[1]

        # model
        h_1 = Dense(128, activation="relu")
        h_2 = Dense(64, activation="relu")
        h_3 = Dense(32, activation="relu")
        s = Dense(1)

        # relevant document score
        rel_doc = Input(shape=(dimension, ), dtype="float32")
        h_1_rel = h_1(rel_doc)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # irrelevant document score
        irr_doc = Input(shape=(dimension, ), dtype="float32")
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
        x1, y1 = features[:-1], labels[:-1]
        x2, y2 = features[1:], labels[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            rank_scores = y1 / (y1 + y2)
            rank_scores[rank_scores == np.inf] = 0.5
            rank_scores = np.nan_to_num(rank_scores, copy=False)

        # train
        model.fit([x1, x2], rank_scores, batch_size=batches, epochs=epochs, verbose=1)

        self._model = model

        # generate scores from document/query features
        self._score_function = backend.function([rel_doc], [rel_score])

    def load(self, path):
        from tensorflow.python.keras.models import load_model
        logger.info('Trained model file found, loading model...')
        self._model = load_model(path)
        self._is_ready = True
        logger.info(' Trained model loaded.')

    def save(self, path):
        logger.info('Saving model ...')
        self._model.save(path)
        logger.info('Model saved to {}.'.format(path))

    def ndcg(self, y_true, y_score, k=10):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = sum((2 ** y_true_sorted[i] - 1.) / np.log2(i + 2) for i in range(k))

        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]

        dcg += sum((2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2) for i in range(k))

        # return ndcg
        return 1 if ideal_dcg == 0 else float(dcg) / float(ideal_dcg)

    def train(self, features, labels, train_ratio=1, epochs=3, batches=10):
        logger.info('Start training model...')
        start_time = time.clock()
        assert isinstance(features, np.ndarray) and isinstance(labels, np.ndarray), \
            'Training data should be in the form of numpy.ndarray'
        assert features.shape[0] == labels.shape[0] and labels.shape[1] == 1, \
            'Feature array and label array mismatch, features: {} and labels: {}'.format(features.shape, labels.shape)
        train_len = int(len(features) * train_ratio)
        train_features = features[:train_len]
        train_labels = labels[:train_len]
        test_features = features[train_len:]
        test_labels = labels[train_len:]

        self._train_model(train_features, train_labels, epochs=epochs, batches=batches)
        self._is_ready = True
        end_time = time.clock()
        logger.info('Model trained in {:.1f} seconds'.format(end_time - start_time))

        logger.info('Start testing...')

        ndcg = 0
        for _ in range(1000):
            # pick 10 items from
            indices = np.random.randint(0, test_features.shape[0], 10)
            to_rank_features = test_features[indices]
            to_rank_labels = test_labels[indices]

            scores = self._score_function([to_rank_features])[0]
            ndcg += self.ndcg(np.array(to_rank_labels),  np.array(scores))

        logger.info('Test ended with NDCG {:.4f}'.format(ndcg / 1000.0))
        return ndcg / 1000.0

    def rank(self, features):
        if not self._is_ready:
            logger.error('Ranker isn\'t ready, train the model or load the pre-trained model first.')
            return None

        logger.info('Start ranking the features with size {}.'.format(len(features)))
        labels = self._score_function([features])[0]
        logger.info('Rank finished.')
        return labels
