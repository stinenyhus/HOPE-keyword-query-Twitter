'''
Counting number of tweets in the relevant data paths
'''

import ndjson
from glob import glob
import os
import re
import datetime


def count_posts(path):
    count = 0
    min_date, max_date = None, None
    for i, in_file in enumerate(glob(path)):
        # print(in_file)
        try:
            date = re.findall(r'\d{8}', in_file)[0]
            mi_date, ma_date = date, date
        except IndexError:
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', in_file)
            mi_date, ma_date = dates[0], dates[1]


        # Checking min and max date
        if not min_date and not max_date:
            min_date, max_date = mi_date, ma_date
        elif mi_date < min_date:
            min_date = mi_date
        elif ma_date > max_date:
            max_date = ma_date

        if i % 10==0:
            print(f'{i}: {in_file}. Date = {mi_date}')
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for _ in reader:
                count += 1
    return count, min_date, max_date


def main(paths: dict, date: str):
    if not os.path.exists("tweet_counts"):
            # print("Dir does not exist")
            os.makedirs("tweet_counts")
    with open(f'tweet_counts/{date}.txt', 'a') as f:
            f.write(f'Counts from {date}: \n\n')
    
    for country, path in paths.items():
        print(country)

        count, min_date, max_date = count_posts(path)
        try:
            start=datetime.datetime.strptime(min_date, "%Y%m%d")
            end=datetime.datetime.strptime(max_date, "%Y%m%d")
        except ValueError:
            start=datetime.datetime.strptime(min_date, "%Y-%m-%d")
            end=datetime.datetime.strptime(max_date, "%Y-%m-%d")
        delta = end-start
        total_days = delta.days

        count_text =f'''{country}:
        In path {path} 
        there were {count} posts. 
        There were {total_days} days, going from {min_date} to {max_date}.
        The average number of tweets per day was {count/total_days}.
        \n------------\n'''
        
        with open(f'tweet_counts/{date}.txt', 'a') as f:
            f.write(count_text)
        
        print(f'>> done with {country}')

if __name__=='__main__':
    paths = {
        'danish': '/data/001_twitter_hope/preprocessed/da/*.ndjson',
        # 'norwegian': '/data/001_twitter_hope/preprocessed/no/*.ndjson',
        # 'swedish': '/data/001_twitter_hope/preprocessed/sv/*.ndjson',
        'english': '/data/twitter-omicron-denmark/*.ndjson'
    }

    main(paths, date='11-05-22')