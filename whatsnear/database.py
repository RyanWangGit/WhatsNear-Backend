# coding=utf-8
import geohash
import sqlite3
from haversine import haversine


def to_str(text):
    try:
        return unicode(text)
    except:
        return str(text)


class Database(object):
    def __init__(self, database):
        self._file_path = database
        self._conn = sqlite3.connect(database)
        self._total_num = 0
        self._categories = {}
        self._get_globals()

    def _get_globals(self):
        self._total_num = int(self._conn.execute('''SELECT COUNT(*) FROM \'Beijing-Checkins\' ''').fetchone()[0])
        for row in self._conn.execute('''SELECT category, COUNT(*) AS num FROM "Beijing-Checkins" GROUP BY category'''):
            self._categories[to_str(row[0])] = int(row[1])
        #for row in self._conn.execute('''SELECT category, COUNT(*) AS num FROM (SELECT category FROM "Beijing-CHeckins" LIMIT 10000) GROUP BY category'''):
            #self._categories[unicode(row[0])] = int(row[1])

    def get_total_num(self):
        return self._total_num

    def get_categories(self):
        return self._categories

    def get_neighboring_points(self, lng, lat, r, geo=None):
        geo_hash = geo[:6] if geo is not None else geohash.encode(float(lat), float(lng), 6)
        neighbors = []
        potential_neighbors = []

        for point in self._conn.execute('''SELECT lat,lng,category,checkins,id FROM \'Beijing-Checkins\' 
                                             WHERE geohash LIKE \'%s%%\'''' % geo_hash):
            potential_neighbors.append(point)

        for neighbor in potential_neighbors:
            if haversine((float(neighbor[0]), float(neighbor[1])), (float(lat), float(lng))) * 1000 <= r:
                neighbors.append({
                    'id': int(neighbor[4]),
                    'lat': float(neighbor[0]),
                    'lng': float(neighbor[1]),
                    'category': to_str(neighbor[2]),
                    'checkins': int(neighbor[3])
                })

        return neighbors

    def expand_info(self, point):
        row = self._conn.execute('''SELECT lng,lat,name,address,category,checkins 
                                      FROM \'Beijing-Checkins\' WHERE id=? LIMIT 1''',
                                 (point['id'],)).fetchone()
        point['lng'] = float(row[0])
        point['lat'] = float(row[1])
        point['name'] = to_str(row[2])
        point['address'] = to_str(row[3])
        point['category'] = to_str(row[4])
        point['checkins'] = int(row[5])
        return point

    def update_geohash(self):
        c = self._conn.cursor()
        # if geohash has never been calculated
        if c.execute('''SELECT geohash FROM \'Beijing-Checkins\' LIMIT 1''').fetchone()[0] is None:
            # calculate the geohash value and store in database
            for row in c.execute('''SELECT id,lng,lat FROM \'Beijing-Checkins\''''):
                self._conn.execute('''UPDATE \'Beijing-Checkins\' set geohash=? WHERE id=?''',
                                   (geohash.encode(float(row[2]), float(row[1])), row[0]))
            self._conn.commit()

    def expand_neighbors(self, point):
        for neighbor in point['neighbors']:
            row = self._conn.execute('''SELECT checkins,category FROM \'Beijing-Checkins\' WHERE id=? LIMIT 1''',
                                     (neighbor['id'],)).fetchone()
            neighbor['checkins'] = int(row[0])
            neighbor['category'] = unicode(row[1])

        return point

    def release_neighbors(self, point):
        for neighbor in point['neighbors']:
            del neighbor['checkins']
            del neighbor['category']

    def get_connection(self):
        return self._conn

    def get_file_path(self):
        return self._file_path

