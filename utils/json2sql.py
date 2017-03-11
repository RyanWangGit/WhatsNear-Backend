# -*- coding: UTF-8 -*-
import codecs
import json
import os
import optparse


def json2sql(json_path, sql_path, table_name, encoding='utf-8', is_multi_line=False):

    # read the data
    data = []
    with codecs.open(json_path, 'r', encoding=encoding) as f:
        if is_multi_line:
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.loads(f)

    # get the keys to creat the table headers
    keys = list(data[0].keys())

    # open the output file
    out_file = codecs.open(sql_path, 'w', encoding=encoding)

    # write the create table
    out_file.write('CREATE TABLE IF NOT EXISTS `%s` (\n' % table_name)
    out_file.write('\t`TABLE_ID` int(11) NOT NULL auto_increment,\n')
    for key in keys:
        out_file.write('\t`%s` varchar(128) default NULL,\n' % key)
    out_file.write('\tPRIMARY KEY (`TABLE_ID`)\n')
    out_file.write(')DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;\n\n')

    # write insert statement
    for item in data:
        out_file.write('INSERT INTO %s(' % table_name)
        for i in range(len(keys)):
            if i == len(keys) - 1:
                out_file.write('%s' % keys[i])
            else:
                out_file.write('%s, ' % keys[i])

        out_file.write(') VALUES (')

        for i in range(len(keys)):
            if i == len(keys) - 1:
                out_file.write('\"%s\"' % item[keys[i]])
            else:
                out_file.write('\"%s\", ' % item[keys[i]])

        out_file.write(');\n')


def main():
    # set up option parser
    parser = optparse.OptionParser()
    parser.add_option('-t', '--table',
                      action='store', dest='table_name', default='TABLE',
                      help='The name of the table to be created or inserted.')
    parser.add_option('-j', '--json',
                      action='store', dest='json_path',
                      help='The path of the json file.')
    parser.add_option('-s', '--sql',
                      action='store', dest='sql_path',
                      help='The path of the output sql file.')
    parser.add_option('-l', '--lines',
                      action='store_true', dest='is_multi_line', default=False,
                      help='Whether the file contains many objects line by line or a big json object.')
    parser.add_option('-e', '--encoding',
                      action='store', dest='encoding', default='utf-8',
                      help='The encoding of the json and sql file.')

    (options, args) = parser.parse_args()

    if not options.json_path:
        print('Missing json file path, set with -j\n')
        parser.print_help()
        exit(-1)

    if not options.sql_path:
        (root, ext) = os.path.splitext(options.json_path)
        options.sql_path = root + '.sql'

    json2sql(options.json_path, options.sql_path, options.table_name,
             encoding=options.encoding, is_multi_line=options.is_multi_line)

if __name__ == '__main__':
    main()
