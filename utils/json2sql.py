# -*- coding: UTF-8 -*-
import codecs
import json
import os
import optparse

# set up option parser
parser = optparse.OptionParser()
parser.set_default('table_name', 'TABLE')
parser.set_default('json_path', './data.json')
parser.set_default('sql_path', '')
parser.add_option('-t', '--table',
                  action='store', type='string', dest='table_name',
                  help='The name of the table to be created or inserted.')
parser.add_option('-j', '--json',
                  action='store', type='string', dest='json_path',
                  help='The path of the json file.')
parser.add_option('-s', '--sql',
                  action='store', type='string', dest='sql_path',
                  help='The path of the output sql file.')

(options, args) = parser.parse_args()

table_name = options.table_name

json_path = options.json_path

sql_path = ''

if options.sql_path == '':
    (root, ext) = os.path.splitext(json_path)
    sql_path = root + '.sql'

# read the data
data = []
with codecs.open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# get the keys to creat the table headers
keys = list(data[0].keys())

# open the output file
out_file = codecs.open(sql_path, 'w', encoding='utf-8')

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
