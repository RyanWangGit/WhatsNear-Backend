# -*- coding: utf-8 -*-
def csvreader(file_path):
    import csv

    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, restkey=None, restval=None)
        out = [row for row in reader]

    return out
