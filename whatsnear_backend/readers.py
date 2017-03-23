# -*- coding: utf-8 -*-
def csvreader(file_path, encoding='utf-8'):
    import csv
    import codecs

    with codecs.open(file_path, 'r', encoding=encoding) as csv_file:
        csv_dict = csv.DictReader(csv_file, restkey=None, restval=None)
        out = [obj for obj in csv_dict]

    return out
