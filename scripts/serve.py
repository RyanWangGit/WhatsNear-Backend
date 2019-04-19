import tornado.ioloop
import tornado.web
import json
from ranknear.ranknet import RankNet
from ranknear.database import Database

# global ranknet object
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
        neighbors = conn.get_neighboring_points(lng, lat, 200)

        self.write(json.dumps(neighbors))


def start_server(database, model, ip='127.0.0.1', port=8080):
    global conn
    global ranknet

    conn = Database(database)
    ranknet.load(model)
    # start hosting the server
    app = tornado.web.Application([
        ('/', WhatsNearHandler),
        ('/query', QueryHandler),
        ('/hot', HotHandler),
        ('/neighbor', NeighborHandler)
    ])

    app.listen(port, ip)

    tornado.ioloop.IOLoop.current().start()
