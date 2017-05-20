import time
import json
import math
from database import Database


class Dataset(object):
    def __init__(self):
        # training related variables
        self._labels = []
        self._features = []

        # global parameters
        self._mean_category_number = {}
        self._category_coefficient = {}
        self._categories = {}
        self._database = None
        self.is_ready = False

    def vectorize_point(self, neighbors, training_category):
        neighbor_categories = self._neighbor_categories(neighbors)

        x = []

        # density
        x.append(len(neighbors))

        # neighbors entropy
        entropy = 0
        for (key, value) in neighbor_categories.items():
            if value == 0:
                continue
            entropy += (float(value) / len(neighbors)) * -1 * math.log(float(value) / len(neighbors))

        x.append(entropy)

        # competitiveness
        competitiveness = 0
        if training_category in neighbor_categories:
            competitiveness = -1 * float(neighbor_categories[training_category]) / len(neighbors)

        x.append(competitiveness)

        # quality by jensen
        jenson_quality = 0
        for category, _ in self._categories.items():
            if self._category_coefficient[category][training_category] == 0:
                continue

            jenson_quality += math.log(self._category_coefficient[category][training_category]) * (
                neighbor_categories[category] - self._mean_category_number[category][training_category])

        x.append(jenson_quality)

        # area popularity
        popularity = 0
        for neighbor in neighbors:
            popularity += int(neighbor['checkins'])

        x.append(popularity)
        return x

    def load(self, path):
        print('[Dataset] Pre-calculated train file found, loading from external file...')
        start_time = time.clock()

        with open(path, 'r') as f:
            self._mean_category_number = json.loads(f.readline())
            self._category_coefficient = json.loads(f.readline())
            self._categories = json.loads(f.readline())
            self._labels = json.loads(f.readline())
            self._features = json.loads(f.readline())

        end_time = time.clock()
        print('[Dataset] Training data read in %f seconds.' % (end_time - start_time))

    def save(self, path):
        with open(path, 'w') as f:
            f.write(json.dumps(self._mean_category_number) + '\n')
            f.write(json.dumps(self._category_coefficient) + '\n')
            f.write(json.dumps(self._categories) + '\n')
            f.write(json.dumps(self._labels) + '\n')
            f.write(json.dumps(self._features))
        print('[Dataset] Calculated training data stored into %s.' % path)

    def get_features(self):
        return self._features

    def get_labels(self):
        return self._labels

    def _neighbor_categories(self, neighbors):
        # calculate sub-global parameters
        neighbor_categories = {}
        for category, _ in self._categories.items():
            neighbor_categories[category] = 0
        for neighbor in neighbors:
            neighbor_categories[neighbor['category']] += 1
        return neighbor_categories

    def prepare(self, database, train_file):
        from progress.bar import Bar
        import multiprocessing as mp

        print('[Dataset] Pre-calculated train file not found, calculating training data...')
        start_time = time.clock()

        self._database.update_geohash()

        # radius to calculate features
        r = 200

        # calculate global parameters
        total_num = self._database.get_total_num()
        # total_num = 10000
        self._categories = self._database.get_categories()

        # calculate and store the neighboring points

        # bar = Bar('Calculating neighbors', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        # for row in self._database.get_connection().execute('''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
        #    self._all_points.append({
        #        'id': int(row[5]),
        #        'checkins': int(row[4]),
        #        'category': unicode(row[3]),
        #        'neighbors': self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
        #    })
        #    bar.next()
        # bar.finish()


        # calculate global category parameters
        # initialize the matrix
        for outer, _ in self._categories.items():
            self._mean_category_number[outer] = {}
            self._category_coefficient[outer] = {}
            for inner, _ in self._categories.items():
                self._mean_category_number[outer][inner] = 0
                self._category_coefficient[outer][inner] = 0

        # split datasets into different parts
        parts = []
        step = total_num / mp.cpu_count()
        for i in xrange(0, total_num, step):
            if i + step < total_num:
                parts.append([i, step])
            else:
                parts.append([i, total_num - i])

        # initialize the result_queue
        result_queue = mp.Queue()
        progress_queue = mp.Queue()

        def progress(title, max, progress_queue):
            bar = Bar(title, suffix='%(index)d / %(max)d, %(percent)d%%', max=max)
            cur = 0
            while cur != max:
                progress = progress_queue.get()
                for _ in range(progress):
                    bar.next()
                cur += progress

        def calculate_mean_category(database, part, categories, result_queue, progress_queue):
            # initialize local matrix
            mean_category_number = {}
            for outer, _ in categories.items():
                mean_category_number[outer] = {}
                for inner, _ in categories.items():
                    mean_category_number[outer][inner] = 0

            database = Database(database)
            # calculate mean category numbers
            bar = Bar('Calculating mean category numbers', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
            for row in database.get_connection().execute(
                            '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\' LIMIT %d,%d''' % (
                    part[0], part[1])):
                neighbors = database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
                for neighbor in neighbors:
                    mean_category_number[neighbor['category']][unicode(row[3])] += 1
                progress_queue.put(1)

            for p, _ in categories.items():
                for l, _ in categories.items():
                    # TODO: to delete this line of code when we run training in full dataset
                    if categories[l] == 0:
                        continue
                    mean_category_number[p][l] /= categories[l]

            result_queue.put(mean_category_number)

            bar.finish()

        progress_process = mp.Process(target=progress,
                                      args=('Calculating mean category numbers', total_num, progress_queue))
        progress_process.start()

        processes = []
        for i in range(mp.cpu_count()):
            process = mp.Process(target=calculate_mean_category, args=(
            self._database.get_file_path(), parts[i], self._categories, result_queue, progress_queue))
            processes.append(process)
            process.start()

        print('[RankNet] Starting %d processes.' % mp.cpu_count())

        progress_process.join()

        # calculate category coefficients
        bar = Bar('Calculating category coefficients', suffix='%(index)d / %(max)d, %(percent)d%%',
                  max=len(self._categories) * len(self._categories))
        for p, _ in self._categories.items():
            for l, _ in self._categories.items():
                # TODO: delete this line of code in full dataset
                if self._categories[p] * self._categories[l] == 0:
                    continue

                k_prefix = float(total_num - self._categories[p]) / (self._categories[p] * self._categories[l])

                k_suffix = 0
                for row in self._database.get_connection().execute(
                        '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
                    if unicode(row[3]) == p:
                        neighbors = self._database.get_neighboring_points(float(row[0]), float(row[1]), r,
                                                                          geo=unicode(row[2]))
                        neighbor_categories = self._neighbor_categories(neighbors)

                        if len(neighbors) - neighbor_categories[p] == 0:
                            continue

                        k_suffix += float(neighbor_categories[l]) / (len(neighbors) - neighbor_categories[p])

                self._category_coefficient[p][l] = k_prefix * k_suffix

                bar.next()

        bar.finish()

        bar = Bar('Calculating features', suffix='%(index)d / %(max)d, %(percent)d%%', max=total_num)
        # calculate features
        for row in self._database.get_connection().execute(
                '''SELECT lng,lat,geohash,category,checkins,id FROM \'Beijing-Checkins\''''):
            neighbors = self._database.get_neighboring_points(float(row[0]), float(row[1]), r, geo=unicode(row[2]))
            # add label
            self._labels.append([int(row[4])])
            # add feature
            self._features.append(self._vectorize_point(neighbors, u'生活娱乐'))

            bar.next()

        bar.finish()

        end_time = time.clock()
        print('[RankNet] Training data calculated in %f seconds.' % (end_time - start_time))

        # store calculated train data
        self._write_train_data(os.path.dirname(self._database.get_file_path()) + '/train.txt')
