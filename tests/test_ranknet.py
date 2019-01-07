from ranknear.dataset import Dataset
from ranknear.ranknet import RankNet


def test_ranknet():
    dataset = Dataset()
    dataset.load('./data/train.json')
    ranknet = RankNet()
    ndcg = ranknet.train(dataset, 0.8)
    assert ndcg > 0.3
