'''
Counting number of tweets in the relevant data paths
'''

import ndjson
from glob import glob


def count_posts(path):
    count = 0
    for i, in_file in enumerate(glob(path)):
        if i % 10==0:
            print(f'{i}: {in_file}')
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for _ in reader:
                count += 1
    return count


def main(paths: dict, date: str):
    with open(f'tweet_counts/{date}.txt', 'a') as f:
            f.write(f'Counts from {date}: \n\n')
    for country, path in paths.items():
        print(country)

        count = count_posts(path)
        count_text =f'''{country}:
        In path {path} 
        there were {count} posts. 
        \n------------\n'''
        with open(f'tweet_counts/{date}.txt', 'a') as f:
            f.write(count_text)
        
        print(f'>> done with {country}')

if __name__=='__main__':
    paths = {
        'danish': '/data/001_twitter_hope/preprocessed/da/*.ndjson',
        'norwegian': '/data/001_twitter_hope/preprocessed/no/*.ndjson',
        'swedish': '/data/001_twitter_hope/preprocessed/sv/*.ndjson',
        'english': '/data/twitter-omicron-denmark/*.ndjson'
    }

    main(paths, date='11-03-22')