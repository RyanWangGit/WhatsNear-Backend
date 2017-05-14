# coding: utf-8
import tornado.ioloop
import tornado.web
import json
import sqlite3
from ranknet import RankNet

# global ranknet object
ranknet = RankNet()
conn = None

class WhatsNearHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Usage: \n' +
                   '/query - [[lng, lat], [lng, lat] ...]\n' +
                   '/hot')


class QueryHandler(tornado.web.RequestHandler):
    def get(self):
        query_points = json.loads(self.get_argument('points'))
        self.add_header('Content-type', 'application/json')

        result = []
        for point in ranknet.rank(query_points, self):
            result.append({
                'lnglat': point,
                'surroundings': []
            })

        self.write(json.dumps(result))

class HotHandler(tornado.web.RequestHandler):
    def get(self):
        self.add_header('Content-type', 'application/json')

        result = []
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT lng,lat,name,address,checkins FROM 'Beijing-Checkins' WHERE category='生活娱乐' AND checkins > 0 ORDER BY checkins DESC LIMIT 1000''')

        for row in cursor.fetchall():
            result.append({
                'lng': unicode(row[0]),
                'lat': unicode(row[1]),
                'name': unicode(row[2]),
                'address': unicode(row[3]),
                'checkins': unicode(row[4])
            })
        self.write(json.dumps(result))


def start_server(database, train=None, port=8080):
    global conn
    conn = sqlite3.connect(database)

    # train the model
    ranknet.train(database, train)

    # start hosting the server
    app = tornado.web.Application([
        ('/', WhatsNearHandler),
        ('/query', QueryHandler),
        ('/hot', HotHandler)
    ])

    ip = '127.0.0.1'
    app.listen(port, ip)
    print('[WhatsNear] Start hosting at http://%s:%d.' % (ip, port))
    tornado.ioloop.IOLoop.current().start()
