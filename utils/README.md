# Utils

This folder contains utility scripts to process data / transform data etc.

### json2sql.py

Converts `json` to `SQL` statements that create the table and inserts all the data. 

It uses `OptionParser`, type `python json2sql.py --help` to see the usage.

```
Usage: json2sql.py JSON_PATH [options]

Options:
  -h, --help            show this help message and exit
  -t TABLE_NAME, --table=TABLE_NAME
                        The name of the table to be created or inserted.
  -s SQL_PATH, --sql=SQL_PATH
                        The path of the output sql file.
  -l, --lines           Whether the file contains many objects line by line or
                        a big json object.
  -e ENCODING, --encoding=ENCODING
                        The encoding of the json and sql file.
```