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

    for _ in range(edge_num):
        sample = random.sample(nodes, 2)

        sample[0]['value'] += 1
        sample[1]['value'] += 1

        edges.append((sample[0]['index'], sample[1]['index']))

    return nodes, infos, edges
