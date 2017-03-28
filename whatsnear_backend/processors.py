# -*- coding: utf-8 -*-
def filter_preprocessor(points, can_pass):
    return [point for point in points if can_pass(point)]


def sample_preprocessor(points, sample_num):
    import random
    return random.sample(points, sample_num)


def random_processor(points, edge_num):
    import random
    nodes = []
    edges = []
    infos = []

    for index in range(len(points)):
        point = points[index]
        nodes.append({
            'index': index,
            'lng': point['lng'],
            'lat': point['lat'],
            'value': 0
        })
        infos.append({
            'index': index,
            'name': point['name'],
            'address': point['address'],
            'telephone': point['telephone']
        })

    small_sample = random.sample(nodes, int(len(nodes) * 0.1))

    edges_map = {}
    for i in range(edge_num):
        if i < edge_num * 0.01:
            sample = random.sample(nodes, 2)
        else:
            sample = random.sample(small_sample, 2)

        sample[0]['value'] += 1
        sample[1]['value'] += 1

        key = '%d,%d' % (min(sample[0]['index'], sample[1]['index']),
                         max(sample[0]['index'], sample[1]['index']))

        if key in edges_map:
            edges_map[key] += 1
        else:
            edges_map[key] = 1

    for edge in edges_map:
        edge_list = [int(x) for x in edge.split(',')]
        edge_list.append(edges_map[edge])
        edges.append(edge_list)

    return nodes, infos, edges
