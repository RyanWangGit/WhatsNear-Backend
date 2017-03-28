# -*- coding: utf-8 -*-
def console_writer(nodes, infos, edges):
    print('nodes:')
    print(nodes)
    print('edges:')
    print(edges)
    print('infos:')
    print(infos)


def csvwriter(nodes, infos, edges, path):
    if not path.endswith('/'):
        path += '/'

    with open(path + 'nodes.csv', 'w') as file:
        for node in nodes:
            file.write('%d,%s,%s,%d\n' % (node['index'], node['lng'], node['lat'], node['value']))

    with open(path + 'infos.csv', 'w') as file:
        for info in infos:
            file.write('%d,%s,%s,%s\n' % (info['index'], info['name'], info['address'], info['telephone']))

    with open(path + 'edges.csv', 'w') as file:
        for edge in edges:
            file.write('%d,%d,%d\n' % (edge[0], edge[1], edge[2]))


