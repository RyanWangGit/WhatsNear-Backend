# -*- coding: UTF-8 -*-
import codecs
import json
import os


def json2sql(json_path, sql_path, table_name, encoding='utf-8', is_multi_line=False):
    # read the data
    data = []
    with codecs.open(json_path, 'r', encoding=encoding) as f:
        if is_multi_line:
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.loads(f)

    # get keys for generating create table statements
    keys = list(data[0].keys())

    # open the output file
    out_file = codecs.open(sql_path, 'w', encoding=encoding)

    # write the create table
    out_file.write('CREATE TABLE IF NOT EXISTS `%s` (\n' % table_name)
    out_file.write('\t`TABLE_ID` INT(11) NOT NULL AUTO_INCREMENT,\n')
    for key in keys:
        out_file.write('\t`%s` VARCHAR(128) DEFAULT NULL,\n' % key)
    out_file.write('\tPRIMARY KEY (`TABLE_ID`)\n')
    out_file.write(')DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;\n\n')

    # write insert statement
    out_file.write('INSERT INTO %s(%s) VALUES\n' % (table_name, ', '.join(keys)))

    # write multiple value
    str_list = []

    for item in data:
        str_list.append('\t(%s)' % ', '.join(['\"' + unicode(value) + '\"' for value in item.values()]))

    out_file.write(',\n'.join(str_list))
    out_file.write(';')


def main():
    import argparse

    # set up option parser
    parser = argparse.ArgumentParser(description='Convert multiple json-formatted data into sql statements.')
    parser.add_argument('json_path', action='store')
    parser.add_argument('-t', '--table',
                        action='store', dest='table_name', default='TABLE',
                        help='The name of the table to be created or inserted.')
    parser.add_argument('-s', '--sql',
                        action='store', dest='sql_path',
                        help='The path of the output sql file.')
    parser.add_argument('-l', '--lines',
                        action='store_true', dest='is_multi_line', default=False,
                        help='Whether the file contains many objects line by line or a big json object.')
    parser.add_argument('-e', '--encoding',
                        action='store', dest='encoding', default='utf-8',
                        help='The encoding of the json and sql file.')

    results = parser.parse_args()

    if not results.sql_path:
        (root, ext) = os.path.splitext(results.json_path)
        results.sql_path = root + '.sql'

    json2sql(results.json_path, results.sql_path, results.table_name,
             encoding=results.encoding, is_multi_line=results.is_multi_line)


if __name__ == '__main__':
    main()
