from whatsnear import WhatsNear

if __name__ == '__main__':
    backend = WhatsNear()
    import argparse

    # set up argument parser
    parser = argparse.ArgumentParser(description='Backend for WhatsNear.')
    parser.add_argument('-p', '--port',
                        action='store', dest='port', default=80, type=int,
                        help='The port to listen on.', required=False)

    results = parser.parse_args()
    backend.start(results.port)
