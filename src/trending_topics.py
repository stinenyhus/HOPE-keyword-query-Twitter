'''
Finding the words/terms that are changing position to the most extreme extend
I.e., most trending and detrending terms
'''
###################
# Import packages #
###################
import ndjson
from glob import glob
import re
import string
import os
import spacy
from functools import partial
from datetime import datetime
import datetime as dt
import pandas as pd
from collections import Counter
from dateutil.relativedelta import relativedelta

####################
# Define functions #
####################

#### Tweet cleaning functions ####
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = remove_emoji(tweet)
    tweet = re.sub(r'@(\S*)\w', '', tweet) #mentions
    tweet = re.sub(r'#\S*\w', '', tweet) # hashtags
    # Remove URLs
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    tweet = re.sub(url_pattern, '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet

def join_tokens(token_list):
    res = " ".join(token_list)
    return res

def lemmatize_tweet(tweet, spacy_model, tokenizer):
    tweet = clean_tweet(tweet)
    tokenized = tokenizer(tweet).doc.text
    spacy_tokens = spacy_model(tokenized)
    lemmatized_tweet = [t.lemma_ for t in spacy_tokens]
    
    hmm = ['   ','  ',' ','','♂','','❤','','🤷','”', '“', "'", '"', '’']
    lemmatized_tweet = [x for x in lemmatized_tweet if x not in hmm]
    lemmatized_tweet = [name for name in lemmatized_tweet if name.strip()]
    
    return lemmatized_tweet

# Revert week function
def revert_week(week):
    return week.split("-")[1]+"-"+week.split("-")[0]

#### ndjson reader and cleaner --> generators ####
def ndjson_gen(filepath: str, must_include:str = "ndjson"):
    """This function is generator yielding one twitter post incl. metadata at a time

    Args:
        filepath (str): path to where ndjson files with posts are stored
        must_include (str): string that the file must include to be processed. Defaults to "ndjson"

    Yields:
        post (dict): the post incl metadata as a dictionary
    """    
    for in_file in os.listdir(filepath):
        # Check that file is actually ndjson file 
        if must_include in in_file:
            with open(os.path.join(filepath, in_file)) as f:
                reader = ndjson.reader(f)

                for post in reader:
                    yield post

def clean_tokenize_gen(generator, function, trouble_dict, stopwords:list, min_date, field="text"):
    """This function cleans and individual tweets and tokenizes it 

    Args:
        generator (generator): a generator yielding one post at a time
        function (function): function to use for lemmatizing
        stopwords (list): list of stopwords to exclude from analysis
        field (str, optional): name of field where the actual tweet is stored. Defaults to "text".

    Yields:
        lemmas, week (tuple of (list, string)): tuple with the list of lemmas and the associated week

    """    
    for post in generator:
        if len(post) < 1:
            continue
        if "^RT" in post[field] or "rt" in post[field]:
            continue
        
        date = post["created_at"]
        # Dates have different formats, handling with try except
        try:
            as_date = datetime.strptime(date[:10], '%Y-%m-%d')
        except ValueError:
            as_date_long = datetime.strptime(date, '%a %b %d %H:%M:%S +0000 %Y')
            as_date = datetime.strftime(as_date_long, '%Y-%m-%d')
            as_date = datetime.strptime(as_date, '%Y-%m-%d')

        # If the file has already been processed, skip
        if datetime.date(as_date) < min_date:
            continue

        lemmas = function(post[field])
        lemmas = [lemma for lemma in lemmas if lemma not in stopwords]
        # Manually assigning week to dates around year change
        if str(as_date)[:10] in trouble_dict.keys():
            week = trouble_dict[str(as_date)[:10]]
        else:
            week = datetime.strftime(as_date, '%V-%Y')

        yield (lemmas, week)



if __name__ == "__main__":
    print(f'starting script at {datetime.now()}')
    # List files for running over 
    path = os.path.join("..", "..", "..", "..", "data", "001_twitter_hope", "preprocessed", "da")
    # files = os.listdir(path)

    # Prepare spacy, tokenizer and partial function for lemmatizing
    print("Preparing spacy, functions and dict with trouble weeks")
    sp = spacy.load('da_core_news_lg')
    tokenizer = sp.tokenizer
    stops = open("stop_words.txt","r+")
    stop_words = stops.read().split() + ["rt"]
    lemma_partial = partial(lemmatize_tweet, spacy_model=sp, tokenizer=tokenizer)
    # Look up table for the two weeks crossing a year
    trouble_weeks = {'2020-12-28': '53-2020',
                     '2020-12-29': '53-2020',
                     '2020-12-30': '53-2020',
                     '2020-12-31': '53-2020', 
                     '2021-01-01': '53-2020',
                     '2021-01-02': '53-2020',
                     '2021-01-03': '53-2020',
 
                     '2021-12-27': '52-2021',
                     '2021-12-28': '52-2021',
                     '2021-12-29': '52-2021',
                     '2021-12-30': '52-2021',
                     '2021-12-31': '52-2021',
                     '2022-01-01': '52-2021',
                     '2022-01-02': '52-2021'}

    # print(f'Before starting the function, keys in dict are: {trouble_weeks.keys()}')

    # Checking what has already been processed
    file_path = "term_freq_all.ndjson"
    with open(file_path, 'r') as f:
        data = ndjson.load(f)
    max_week = max([revert_week(week) for week in data[0].keys()])
    year = max_week.split("-")[0]
    minimum_date = dt.date(int(year), 1, 1) + relativedelta(weeks=+int(max_week.split("-")[1]))
    print(f'minimum date is {minimum_date}')

    # Prepare generators
    gen = ndjson_gen(path)
    print('Prepared first generator')
    
    tuple_gen = clean_tokenize_gen(gen, lemma_partial, trouble_weeks, stopwords=stop_words, min_date=minimum_date)
    print('Prepared second generator')

    # Prepare dictionary
    terms_weekly = {}
    start = datetime.now()
    for i,f in enumerate(tuple_gen):
        if f[1] in terms_weekly.keys():
            terms_weekly[f[1]].update(Counter(f[0]))
        else:
            terms_weekly[f[1]] = Counter(f[0])
        # if i == 10:
        #     break
        if i % 5000 == 0:
            print(f'Now processed {i} tweets, total time spent is {datetime.now() - start}')
            print(f'Tweet number {i} has date key {f[1]}')
    
    print("Finished all files, now saving dictionary")
    combined = {**data[0], **terms_weekly}
    with open("term_freq_all.ndjson", "w", encoding='utf8') as f:
        ndjson.dump([combined], f, ensure_ascii=False)




