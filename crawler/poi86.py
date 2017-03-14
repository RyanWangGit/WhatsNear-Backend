# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup
import requests
import json
import codecs
import sys


def crawl_poi86(province_page, output_path):
    root_url = 'http://www.poi86.com'

    # get the province page
    result = requests.get(url=root_url + province_page)
    soup = BeautifulSoup(result.text, 'lxml')

    with codecs.open(output_path, 'w', encoding='utf-8') as out_file:

        # find all district page urls
        for link in soup.find_all('li', class_='list-group-item'):
            print('Crawling ' + link.a.string)

            # navigate to district page
            district_page = root_url + link.a['href']

            # loop for all the items in the district
            page_num = 1
            while True:
                sys.stdout.write(' ' * 10 + '\r')
                sys.stdout.flush()
                print('\t Crawling %d page.' % page_num)

                result = requests.get(url=district_page)
                soup_s = BeautifulSoup(result.text, 'lxml')
                table = soup_s.find('div', class_='panel-body')

                item_count = 1
                for item in table.find_all('tr'):
                    if not item.td:
                        continue

                    # navigate to the specific item page
                    result = requests.get(url=root_url + item.td.a['href'])
                    soup = BeautifulSoup(result.text, 'lxml')

                    # fill the item
                    json_obj = {'name': soup.find('div', class_='panel-heading').h1.string}
                    for group_item in soup.find_all('li', class_='list-group-item'):
                        label, data = group_item.text.split(': ')

                        if label == u'详细地址':
                            json_obj['location'] = data
                        elif label == u'所属分类':
                            json_obj['category'] = data
                        elif label == u'所属标签':
                            json_obj['label'] = data
                        elif label == u'百度坐标':
                            json_obj['lng'], json_obj['lat'] = data.split(',')

                    json.dump(json_obj, out_file, encoding='utf-8', ensure_ascii=False)
                    out_file.write('\n')
                    sys.stdout.write('\t\tCrawled %d items' % item_count + '\r')
                    sys.stdout.flush()
                    item_count += 1

                out_file.flush()

                page_control = soup_s.find('ul', class_='pagination')
                next_page = ''
                end_page = ''
                for li in page_control.find_all('li'):
                    if li.string == u'下一页':
                        next_page = li.a['href']
                    elif li.string == u'尾页':
                        end_page = li.a['href']

                # if we reach the end of the pages
                if next_page == end_page:
                    break
                else:
                    district_page = root_url + next_page

                page_num += 1


def main():
    crawl_poi86('/poi/province/131.html', 'result.txt')

if __name__ == '__main__':
    main()
