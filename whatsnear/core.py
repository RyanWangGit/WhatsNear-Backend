import tornado.ioloop
import tornado.web
import json
import os
from .ranknet import RankNet
from .database import Database
from .dataset import Dataset

# global ranknet object
dataset = Dataset()
ranknet = RankNet()
conn = None

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
                'neighbors': conn.get_neighboring_points(point[1], point[2], 200)
            }

            points.append(new_point)

        ranked_points = ranknet.rank(points, self)

        self.write(json.dumps(ranked_points))


class HotHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Content-type', 'application/json')
        self.add_header('Access-Control-Allow-Origin', '*')

        result = []
        cursor = conn.get_connection().cursor()
        cursor.execute(
            '''SELECT lng,lat,name,address,checkins,id FROM 'Beijing-Checkins' WHERE category='生活娱乐' AND checkins > 0 ORDER BY checkins DESC LIMIT 1000''')

        for row in cursor.fetchall():
            result.append({
                'id': unicode(row[5]),
                'lng': unicode(row[0]),
                'lat': unicode(row[1]),
                'name': unicode(row[2]),
                'address': unicode(row[3]),
                'checkins': int(row[4])
            })
        self.write(json.dumps(result))


class NeighborHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Content-type', 'application/json')
        self.add_header('Access-Control-Allow-Origin', '*')

        lng, lat = json.loads(self.get_argument('point'))
        neighbors = conn.get_neighboring_points(lng, lat, 200)

        self.write(json.dumps(neighbors))


def start_server(database, train=None, model=None, ip='127.0.0.1', port=8080):
    global conn
    global ranknet
    global dataset

    conn = Database(database)
    dirname = os.path.dirname(database)

    if train is not None:
        dataset.load(train)
    else:
        dataset.prepare(database)
        dataset.save(dirname + '/train.txt')

    if model is not None:
        ranknet.load(model)
    else:
        ranknet.train(dataset, 0.8)
        ranknet.save(dirname + '/model.h5')

    # start hosting the server
    app = tornado.web.Application([
        ('/', WhatsNearHandler),
        ('/query', QueryHandler),
        ('/hot', HotHandler),
        ('/neighbor', NeighborHandler)
    ])

    app.listen(port, ip)
    print('[WhatsNear] Start hosting at http://%s:%d.' % (ip, port))
    tornado.ioloop.IOLoop.current().start()
