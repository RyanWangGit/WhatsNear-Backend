# -*- coding: UTF-8 -*-
import codecs
import json


TABLE_NAME = "Gaode"

PATH = 'C:/Users/Ryan/OneDrive/GraduationThesis/data/Zhuhai/gaode/data'
# read the data
data = []
with codecs.open(PATH + '.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# get the keys to creat the table headers
keys = list(data[0].keys())

# open the output file
out_file = codecs.open(PATH + '.sql', 'w', encoding='utf-8')

# write the create table
out_file.write('CREATE TABLE IF NOT EXISTS `%s` (\n' % TABLE_NAME)
out_file.write('\t`TABLE_ID` int(11) NOT NULL auto_increment,\n')
for key in keys:
    out_file.write('\t`%s` varchar(128) default NULL,\n' % key)
out_file.write('\tPRIMARY KEY (`TABLE_ID`)\n')
out_file.write(')DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;\n\n')

# write insert statement
for item in data:
    out_file.write('INSERT INTO %s(' % TABLE_NAME)
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
