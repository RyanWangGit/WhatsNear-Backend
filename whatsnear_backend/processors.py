# -*- coding: utf-8 -*-
def filter_preprocessor(points, can_pass):
    return [point for point in points if can_pass(point)]


def sample_preprocessor(points, sample_num):
    import random
    return random.sample(points, sample_num)


def empty_processor(points):
    nodes = []
    edges = []
    infos = []
    return nodes, infos, edges