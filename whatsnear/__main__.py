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
    parser.add_argument('-f', '--file',
                        action='store', dest='file', type=str,
                        help='The points data file to read from. Must be .csv file', required=True)
    results = parser.parse_args()

    print('[WhatsNear] Loading csv file...')
    start_time = time.clock()

    with open(results.file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, restkey=None, restval=None)
        points = [row for row in reader]

    end_time = time.clock()
    print('[WhatsNear] Csv file loaded in %f seconds, starting server...' % (end_time - start_time))

    # start server
    whatsnear.start_server(points, results.port)
