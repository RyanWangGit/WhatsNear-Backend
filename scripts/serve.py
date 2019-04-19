import tornado.ioloop
import tornado.web
import json
from ranknear.ranknet import RankNet
from ranknear.database import Database

# global ranknet object
ranknet = RankNet()
connection = None


class WhatsNearHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Access-Control-Allow-Origin', '*')
        self.write('Usage: <br />' +
                   '/query - [[lng, lat], [lng, lat] ...] <br />' +
                   '/hot <br />' +
                   '/neighbor [lng, lat]')


class QueryHandler(tornado.web.RequestHandler):
    def get(self):
        query_points = json.loads(self.get_argument('points'))
        self.add_header('Content-type', 'application/json')
        self.add_header('Access-Control-Allow-Origin', '*')

        # pre-process the points
        points = []
        for point in query_points:
            new_point = {
                'id': point[0],
                'lng': point[1],
                'lat': point[2],
                'neighbors': connection.get_neighboring_points(point[1], point[2], 200)
            }

            points.append(new_point)

        ranked_points = ranknet.rank(points)

        self.write(json.dumps(ranked_points))


class HotHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Content-type', 'application/json')
        self.add_header('Access-Control-Allow-Origin', '*')

        result = []
        cursor = connection.get_connection().cursor()
        cursor.execute(
            '''SELECT lng, lat, name, address, checkins, id FROM 'Beijing-Checkins'
                   WHERE category='生活娱乐' AND checkins > 0 ORDER BY checkins DESC LIMIT 1000''')

        for row in cursor.fetchall():
            result.append({
                'id': str(row[5]),
                'lng': str(row[0]),
                'lat': str(row[1]),
                'name': str(row[2]),
                'address': str(row[3]),
                'checkins': int(row[4])
            })
        self.write(json.dumps(result))


class NeighborHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Content-type', 'application/json')
        self.add_header('Access-Control-Allow-Origin', '*')

        lng, lat = json.loads(self.get_argument('point'))
        neighbors = connection.get_neighboring_points(lng, lat, 200)

        self.write(json.dumps(neighbors))


def main():
    global connection
    global ranknet

    # set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Backend for WhatsNear.')
    parser.add_argument('-i', '--ip',
                        action='store', dest='ip', default='127.0.0.1', type=str,
                        help='The ip to bind on.', required=False)
    parser.add_argument('-p', '--port',
                        action='store', dest='port', default=80, type=int,
                        help='The port to listen on.', required=False)
    parser.add_argument('-s', '--sqlite',
                        action='store', dest='sqlite', type=str,
                        help='The SQLite3 database to read from.', required=True)
    parser.add_argument('-m', '--model',
                        action='store', dest='model', type=str,
                        help='The trained model to read from.', required=True)
    results = parser.parse_args()

    # start server
    connection = Database(results.sqlite)
    ranknet.load(results.model)
    # start hosting the server
    app = tornado.web.Application([
        ('/', WhatsNearHandler),
        ('/query', QueryHandler),
        ('/hot', HotHandler),
        ('/neighbor', NeighborHandler)
    ])

    app.listen(results.port, results.ip)

    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
