# coding: utf-8

import csv
import time
import whatsnear

if __name__ == '__main__':
    # set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Backend for WhatsNear.')
    parser.add_argument('-p', '--port',
                        action='store', dest='port', default=80, type=int,
                        help='The port to listen on.', required=False)
    parser.add_argument('-s', '--sqlite',
                        action='store', dest='sqlite', type=str,
                        help='The SQLite3 database to read from.', required=True)
    parser.add_argument('-t', '--train',
                        action='store', dest='train', type=str,
                        help='The training matrix file to read from.', required=False)
    parser.add_argument('-i', '--ip',
                        action='store', dest='ip', type=str,
                        help='The ip to bind on.', required=False)
    results = parser.parse_args()

    # start server
    whatsnear.start_server(results.sqlite, results.train, results.ip, results.port)
