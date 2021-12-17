"""
Retrieve initial stats of data and preprocess
"""

import pandas as pd
from icecream import ic
import getopt, sys, os
import re
import glob

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

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def preprocess_stats(data_prefix: str, 
                     root_path:str,
                     from_date:str, 
                     to_date:str):
    """Main preprocessing, cleans data, masks it if necessary
    data_prefix: indicates which dataset it is
    root_path: location for the input and output
    from_date: date from which 
    """
    # input_data = root_path + data_prefix + "_data.csv"
    input_data = f'{root_path}{data_prefix}_files/{data_prefix}_data.csv'

    df = pd.read_csv(input_data,lineterminator='\n')[1:].rename(columns={"0":"created_at", "1":"id", "2":"text", "3":"search_keyword"})
    print(df.head())

    df = df[df["created_at"] != '0'].reset_index(drop=True)
    df = df[df["created_at"] != 'created_at'].reset_index(drop=True)
    df = df.sort_values(by='created_at').reset_index(drop=True)
    print(len(df))
    
    print(df.created_at.unique())
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')
    
    if (len(from_date) > 1) and (len(to_date) > 1):
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
    
    print(df3.groupby(["search_keyword"]).count().reset_index())
    
    # out_filename = root_path + data_prefix + "_data_pre.csv"
    out_filename = f'{root_path}{data_prefix}_files/{data_prefix}_data_pre.csv'
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
        opts, args = getopt.getopt(argv,"hk:f:t:l:s:")
    except getopt.GetoptError:
        print('test.py -k <keyword1,keyword2> -f <2020-01-20> -t <2020-12-31> -l <20200101>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -keywords <keyword1,keyword2> -from_date <2020-01-20> -to_date <2020-12-31> -test_limit <20200101>')
            sys.exit()
        elif opt in "-k":
            keywords = arg
        elif opt in "-f":
            from_date = arg
            print('Date specifics: from ', from_date)
        elif opt in "-t":
            to_date = arg
            print(' to ', to_date)
        elif opt in "-l":
            test_limit = arg
            print('TESTING: ', test_limit)
        elif opt in "-s":
            small = arg
            print('Small: ', small)
    print('Input keywords are ', keywords)
    return keywords, from_date, to_date

########################################################################################################################
##     INPUT
########################################################################################################################
    
if __name__ == "__main__":
    
    keywords, from_date, to_date = main(sys.argv[1:])
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
    preprocess_stats(data_prefix, root_path, from_date, to_date)