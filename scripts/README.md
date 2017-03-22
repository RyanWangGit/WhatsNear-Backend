# WhatsNear-Backend - Scripts

## json2sql.py (Python 2.7+)

Converts `json` to `SQL` statements that create the table and inserts all the data. 

It uses `ArgumentParser`, type `python json2sql.py --help` to see the usage.

Note that as `ArgumentParser` was introduced into Python standard library since 2.7, 
the script is supposed to run under `Python 2.7+` otherwise you might have to run 
`pip install argparse` to install it.

```
usage: json2sql.py [-h] [-t TABLE_NAME] [-s SQL_PATH] [-l] [-e ENCODING]
                   json_path

Convert multiple json-formatted data into sql statements.

positional arguments:
  json_path

optional arguments:
  -h, --help            show this help message and exit
  -t TABLE_NAME, --table TABLE_NAME
                        The name of the table to be created or inserted.
  -s SQL_PATH, --sql SQL_PATH
                        The path of the output sql file.
  -l, --lines           Whether the file contains many objects line by line or
                        a big json object.
  -e ENCODING, --encoding ENCODING
                        The encoding of the json and sql file.
```

## poi86.py

This script is used to get POI data from [poi86.com](http://www.poi86.com).
