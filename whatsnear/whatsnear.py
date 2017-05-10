import tornado.ioloop
import tornado.web
import json
from ranknet import RankNet

# global ranknet object
ranknet = RankNet()

class WhatsNearHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Usage: \n' +
                   '/query - [[lng, lat], [lng, lat] ...]')


class QueryHandler(tornado.web.RequestHandler):
    def get(self):
        query_points = self.get_argument('points')
        self.add_header('Content-type', 'application/json')
        self.write(json.dumps(ranknet.rank(query_points)))


def start_server(points, port=8080):
    ranknet.load(points)

    app = tornado.web.Application([
        ('/', WhatsNearHandler),
        ('/query', QueryHandler)
    ])

    ip = '127.0.0.1'
    app.listen(port, ip)
    print('[WhatsNear] Start hosting at http://%s:%d.' % (ip, port))
    tornado.ioloop.IOLoop.current().start()
