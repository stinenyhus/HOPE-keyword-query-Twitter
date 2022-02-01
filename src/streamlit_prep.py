'''
Prepares data for streamlit app

The script outputs five .pkl files in the folder '{data_prefix}_files/{data_prefix}_streamlit/' 
'''
####################
# Import libraries #
####################

import pandas as pd
import collections
import itertools
import re 
import ast
from nltk.util import bigrams 
import spacy
import os
import sys
import getopt
import pickle
from configparser import ConfigParser


#####################
# Define functions #
####################

def load_data(data_prefix: str) -> pd.DataFrame:

    data_prefix = data_prefix
    map_prefix = f'{data_prefix}_files'

    filename = '../' + map_prefix + '/' + data_prefix + '_final.csv'
    data = pd.read_csv(filename, lineterminator="\n")
      
    if not os.path.exists('../fig'):
        os.makedirs('../figs')
    if not os.path.exists(f'../fig/{data_prefix}'):
        os.makedirs(f'../fig/{data_prefix}')
    
    return data


def get_tweet_frequencies(df):
    # Add freq of tweets by themselves in the dataset
    # pd.DataFrame creates a new datafrme where "number of tweets" is now the column and it is filled with the associated number
    df_ = df.drop("nr_of_tweets", axis=1)
    tweet_freq = pd.DataFrame({'nr_of_tweets' : df_.groupby(['date']).size()}).reset_index()
    #merge 
    freq_tweets = pd.merge(df_, tweet_freq, how='left', on=['date'])#, 'id', 'created_at'])
    
    return freq_tweets    



def prepare_date_col(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values('created_at')
    data['date'] = pd.to_datetime(data['created_at'], utc=True).dt.strftime('%Y-%m-%d')
    data['date'] = pd.to_datetime(data['date'])

    return data


def extract_hashtags(row):
    unique_hashtag_list = list(re.findall(r'#\S*\w', row['text']))
    return unique_hashtag_list


def hashtag_per_row(data):
    # Create hashtags column with the actual unique hashtags
    data['hashtags'] = data.apply(lambda row: extract_hashtags(row), axis = 1)

    # Let's take a subset of necessary columns, add id
    df = data[['date', 'hashtags']].reset_index().rename(columns={'index': 'id'})

    # Select only the ones where we have more than 1 hashtag per tweet
    df = df[df['hashtags'].map(len) > 1].reset_index(drop=True)

    # Hashtag per row
    # convert list of pd.Series then stack it
    df = (df
     .set_index(['date','id'])['hashtags']
     .apply(pd.Series)
     .stack()
     .reset_index()
     .drop('level_2', axis=1)
     .rename(columns={0:'hashtag'}))
    #lowercase!
    df['hashtag'] = df['hashtag'].str.lower()
    df['hashtag'] = df['hashtag'].str.replace("'.", "")
    df['hashtag'] = df['hashtag'].str.replace("’.", "")

    return df

# Aggregate a frequency DF
def get_hashtag_frequencies(df):
    # Add freq of hashtags by themselves in the dataset
    tweet_freq = pd.DataFrame({'nr_of_hashtags' : df.groupby(['hashtag']).size()}).reset_index()
    return tweet_freq


# Calculate word frequency
def word_freq(data: pd.DataFrame):
    w_freq = data.tokens_string.str.split(expand = True).stack().value_counts()
    w_freq = w_freq.to_frame().reset_index().rename(columns={'index': 'word', 0: 'Frequency'})
    df_freq= w_freq.nlargest(30, columns=['Frequency'])

    return df_freq



# Bigrams
def get_bigrams(data: pd.DataFrame) -> pd.DataFrame:
    terms_bigram = [list(bigrams(ast.literal_eval(tweet))) for tweet in data['tokens_list']]
    # Flatten list of bigrams in clean tweets
    bigrms = list(itertools.chain(*terms_bigram))
    
    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrms)
    bigram_df = pd.DataFrame(bigram_counts.most_common(30), columns=['bigram', 'count'])

    return bigram_df

# wordcloud preparation
def wordcloud_prep(data: pd.DataFrame) -> pd.DataFrame:
    sp = spacy.load('da_core_news_lg')
    file = open("stop_words.txt","r+")
    stop_words = file.read().split()
    # Lemmatize stop words
    stops = " ".join(stop_words)
    stops = sp(stops)
    my_stop_words = [t.lemma_ for t in stops]

    texts = data["tokens_string"]
    texts = texts[texts.notnull()] #Sometimes necessary
    texts = ", ".join(texts)

    return texts, my_stop_words


# Save pkl files in a folder
def save_file(data: pd.DataFrame,
              path: str,
              file_name: str):

    if not os.path.exists(path):
        os.mkdir(path)

    file_path = os.path.join(path, file_name) 
    
    if os.path.exists(file_path):
        print('File already exists!')
    
    print(f'Saving {file_path} \n---------------\n')
    data.to_pickle(file_path)

def save_dict(data: dict,
              path: str,
              file_name: str):

    if not os.path.exists(path):
        os.mkdir(path)

    file_path = os.path.join(path, file_name) 
    
    if os.path.exists(file_path):
        print('File already exists!')
    
    print(f'Saving {file_path} \n---------------\n')
    with open(file_path, 'wb') as handle:
        pickle.dump(wordcloud_dict, handle)


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
            small = config[f'{key}']["small"]
            print(f'Running streamlit preparation with key: {key}, keywords: {keywords} from {from_date} and small = {small}')
    return keywords, from_date, to_date, small


if __name__ == "__main__":

    keywords, from_date, to_date, small = main(sys.argv[1:])
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

    print('--STREAMLIT PREPARATION--')
    print('Data prefix = ', data_prefix, '\n---------------\n')
    ## Load file ##
    df = load_data(data_prefix=data_prefix)

    ## make .pkl files ##
    out_path = os.path.join('..', f'{data_prefix}_files', f'{data_prefix}_streamlit')

    df = prepare_date_col(df)
    df = get_tweet_frequencies(df)
    save_file(df, out_path, f'{data_prefix}.pkl')

    hashtags = hashtag_per_row(df)
    freq_hashtags = get_hashtag_frequencies(hashtags)
    df1 = freq_hashtags.sort_values(by=['nr_of_hashtags'], ascending=False)
    hashtags = df1.nlargest(30, columns=['nr_of_hashtags'])
    save_file(hashtags, out_path, f'{data_prefix}_hash.pkl')

    w_freqs = word_freq(df)
    save_file(w_freqs, out_path, f'{data_prefix}_w_freq.pkl')

    bigrams = get_bigrams(df) 
    save_file(bigrams, out_path, f'{data_prefix}_bigrams.pkl')

    texts, stop_words = wordcloud_prep(df)
    wordcloud_dict = {'texts': texts, 'my_stop_words': stop_words}
    save_dict(wordcloud_dict, out_path, f'{data_prefix}_wordcloud.pkl')
