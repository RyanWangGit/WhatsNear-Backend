# -*- coding: utf-8 -*-
from deprecated import whatsnear_backend as readers


def test_csvreader():

    out = readers.csvreader('./tests/data/test_original.csv')
    print(out)
    assert out == [{
        'col1': '115.491835',
        'col2': '39.997465',
        'col3': 'row1 1',
        'col4': 'row1 2'
    }, {
        'col1': '115.491888',
        'col2': '39.998826',
        'col3': '行2_1',
        'col4': '行2_2'
    }]
