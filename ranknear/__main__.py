import ranknear


def main():
    # set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='RankNear - Rank the locations nearby using RankNet.')
    parser.add_argument('-s', '--sqlite',
                        action='store', dest='sqlite', type=str,
                        help='The SQLite3 database to read from.', required=True)
    parser.add_argument('-t', '--train',
                        action='store', dest='train', type=str,
                        help='The training matrix file to read from.', required=False)
    parser.add_argument('-o', '--out',
                        action='store', dest='model', type=str,
                        help='The model file to output.', default='./model', required=False)
    results = parser.parse_args()

    # train the model
    database = ranknear.Database(results.sqlite)
    dataset = ranknear.Dataset()
    if results.train:
        dataset.load(results.train)
    else:
        dataset.prepare(database)

    ranknet = ranknear.RankNet()
    ranknet.train(dataset.get_features(), dataset.get_labels())
    ranknet.save(results.out)


if __name__ == '__main__':
    main()
