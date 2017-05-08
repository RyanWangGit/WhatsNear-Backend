import whatsnear

if __name__ == '__main__':
    # set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Backend for WhatsNear.')
    parser.add_argument('-p', '--port',
                        action='store', dest='port', default=80, type=int,
                        help='The port to listen on.', required=False)
    results = parser.parse_args()

    # start server
    whatsnear.start_server(results.port)
