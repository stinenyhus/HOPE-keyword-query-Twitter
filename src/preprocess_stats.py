"""
Retrieve initial stats of data and preprocess
"""

import pandas as pd
from icecream import ic
import getopt, sys, os
import re
import glob
from configparser import ConfigParser
from ast import literal_eval
from datetime import date
from functools import partial

########################################################################################################################
##     DEFINE FUNCTIONS
########################################################################################################################

def remove_emoji(string:str):
    """Remove all emojis (captures a lot but not everything)
    string: str
    """
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

def remove_mentions(row):
    """Remove mentions, hashtags, URLs, emojis
    row: pandas DataFrame row with column "text"
    """
    tweet = row["text"]
    clean_tweet = re.sub(r'@(\S*)\w', '', tweet) #mentions
    clean_tweet = re.sub(r'#\S*\w', '', clean_tweet) # hashtags
    # Remove URLs
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    clean_tweet = re.sub(url_pattern, '', clean_tweet)
    clean_tweet = remove_emoji(clean_tweet)
    return clean_tweet

def remove_quote_tweets(df):
    """Creates mentioneless_text, remove bot-like tweets/quote tweets, when 50 first characters are the exact same, remove as duplicates
    df: pandas DataFrame with column "text"
    """
    df["text"] = df["text"].astype(str)
    df["mentioneless_text"] = df.apply(lambda row: remove_mentions(row), axis = 1)
    # print("Generated mentioneless texts")
    df["text50"] = df["mentioneless_text"].str[0:50]
    
    df["dupe50"] = df["text50"].duplicated(keep = "first")

    # print("Length of quote tweets: ")
    # ic(len(df[df["dupe50"] == True]))
    
    df = df[df["dupe50"] == False].reset_index()
    return df

# Aggregate a frequency DF
def get_tweet_frequencies(df):
    """Get tweet frequency, how many tweets per day
    df: pandas DataFrame with column "date"
    """
    # Add freq of hashtags by themselves in the dataset
    tweet_freq = pd.DataFrame({'nr_of_tweets' : df.groupby(['date']).size()}).reset_index()
    # Add the whole_frew to id_hashtag
    freq_tweets = pd.merge(df, tweet_freq, how='left', on=['date'])#, 'id', 'created_at'])
    return freq_tweets

# Check whether the searched for keywords include specific words
def check_keywords(string: str, lst1: list, lst2: list):
    check = all([any([x in string for x in lst1]), any([x in string for x in lst2])])
    return check

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def preprocess_stats(data_prefix: str, 
                     root_path:str,
                     from_date:str, 
                     to_date:str, 
                     language: str):
    """Main preprocessing, cleans data, masks it if necessary
    data_prefix: indicates which dataset it is
    root_path: location for the input and output
    from_date: date from which
    language: specifies whether it is from the danish or english tweets (da and en respectively) 
    """
    # input_data = root_path + data_prefix + "_data.csv"
    input_data = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_data.csv')

    if language == 'en':
        df = pd.read_csv(input_data,lineterminator='\n', index_col=0).dropna()
    else:
        df = pd.read_csv(input_data,lineterminator='\n')[1:].rename(columns={"0":"created_at", "1":"id", "2":"text", "3":"search_keyword"})
    
    if data_prefix == "danmark":
        # If the query is denmark talking about covid in denmark
        # Then make sure to only include relevant tweets, not tweets on only either dk or covid
        dk = ["dk", "danmark"]
        cov = ["corona", "covid", "omicron", "omikron"]
        partial_func = partial(check_keywords, lst1=dk, lst2=cov)
        df["relevant"] = list(map(partial_func, df["search_keyword"]))
        df = df[df["relevant"]==True]
        df = df.drop(columns=["relevant"])

    print(df.head())

    df = df[df["created_at"] != '0'].reset_index(drop=True)
    df = df[df["created_at"] != 'created_at'].reset_index(drop=True)
    df = df.sort_values(by='created_at').reset_index(drop=True)
    print(len(df))
     
    print(df.created_at.unique())
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')
    
    if language == 'da' and from_date and to_date:
        # Choose specified date range
        mask = (df['date'] > from_date) & (df['date'] <= to_date)
        df = df.loc[mask]
    elif from_date:
        mask = (df['date'] > from_date)
        df = df.loc[mask]

    
    print("\nStart removing quote tweets")
    df2 = remove_quote_tweets(df)
    print("\nGet tweet frequencies")
    df3 = get_tweet_frequencies(df2)
    
    if language == 'da':
        print(df3.groupby(["search_keyword"]).count().reset_index())
    
    # out_filename = root_path + data_prefix + "_data_pre.csv"
    out_filename = os.path.join(root_path, f'{data_prefix}_files',f'{data_prefix}_data_pre.csv')
    df3.to_csv(out_filename, index=False)
    
########################################################################################################################
##     DEFINE INPUT
########################################################################################################################

def main(argv):
    keywords = ''
    from_date = '' 
    to_date = ''
    test_limit = '' # this is so that it's possible to test the system on just one day/month of data
    small = ''
    try: 
        opts, args = getopt.getopt(argv,"hk:")
        config = ConfigParser()
        config.read("keyword_config.ini")
        
    except getopt.GetoptError:
        print('test.py -k <keyword1,keyword2> -f <2020-01-20> -t <2020-12-31> -l <20200101>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -keywords <keyword1,keyword2> -from_date <2020-01-20> -to_date <2020-12-31> -test_limit <20200101>')
            sys.exit()
        elif opt in "-k":
            key = arg
            keywords = config[f'{key}']["keywords"]
            from_date = config[f'{key}']["from_date"]
            to_date = config[f'{key}']["to_date"]
            test_limit = config[f'{key}']["test_limit"]
            small = literal_eval(config[f'{key}']["small"])
            language = config[f'{key}']["lan"]
            print(f'Running preprocessing with key: {key}, keywords: {keywords} from {from_date}. Small = {small}. Language = {language}.')

    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    to_date = date.today().strftime("%Y-%m-%d")
    test_limit = None if test_limit == 'None' else test_limit
    
    return keywords, from_date, to_date, language

########################################################################################################################
##     INPUT
########################################################################################################################
    
if __name__ == "__main__":
    
    keywords, from_date, to_date, language = main(sys.argv[1:])
    ori_keyword_list = keywords.split(",")
    
    keyword_list = []
    for keyword in ori_keyword_list:
        if re.findall("~#", keyword):
            keyword = re.sub('~', '', keyword)
        else:
            keyword = re.sub("~", " ", keyword)
        keyword_list.append(keyword)
    
    print(keyword_list)

    data_prefix = keyword_list[0]
    # root_path = "/home/commando/stine-sara/HOPE-keyword-query-Twitter/"
    root_path = os.path.join("..") 
    
    ############################
    print("---PREPROCESS STATS---")
    preprocess_stats(data_prefix, root_path, from_date, to_date, language)
