from ranknear.dataset import Dataset
from ranknear.ranknet import RankNet
import numpy as np


def test_ranknet():
    dataset = Dataset()
    dataset.load('./data/train.json')
    ranknet = RankNet()
    ndcg = ranknet.train(np.array(dataset.get_features()), np.array(dataset.get_labels()), 0.8)
    assert ndcg > 0.3
