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
from wordcloud import STOPWORDS
import os
import sys
import getopt
import pickle
from typing import List
from configparser import ConfigParser
from ast import literal_eval
from fileinput import filename
import pandas as pd 
import numpy as np
import seaborn as sns
import os
import sys


#####################
# Define functions #
####################

def load_data(data_prefix: str):
    """Loads the df called _final into pd.DataFrame

    Args:
        data_prefix (str): the keyword prefix

    Returns:
        pd.DataFrame: the _final.csv file as pd.DataFrame
    """    

    data_prefix = data_prefix
    map_prefix = f'{data_prefix}_files'

    filename = '../' + map_prefix + '/' + data_prefix + '_final.csv'
    data = pd.read_csv(filename, lineterminator="\n")
      
    if not os.path.exists('../fig'):
        os.makedirs('../figs')
    if not os.path.exists(f'../fig/{data_prefix}'):
        os.makedirs(f'../fig/{data_prefix}')
    
    return data


def get_tweet_frequencies(df: pd.DataFrame):
    """
    Add freq of tweets by themselves in the dataset

    Args:
        df (pd.DataFrame): dataframe with columns nr_of_tweets and date

    Returns: 
        freq_tweets (pd.DataFrame): dataframe with nr of tweets by day
    """
    # pd.DataFrame creates a new datafrme where "number of tweets" is now the column and it is filled with the associated number
    df_ = df.drop("nr_of_tweets", axis=1)
    tweet_freq = pd.DataFrame({'nr_of_tweets' : df_.groupby(['date']).size()}).reset_index()
    #merge 
    freq_tweets = pd.merge(df_, tweet_freq, how='left', on=['date'])#, 'id', 'created_at'])
    
    return freq_tweets    



def prepare_date_col(data: pd.DataFrame):
    """
    Prepocessing date column in provided df

    Args:
        data (pd.DataFrame): dataframe with column created_at
    
    Returns: 
        data (pd.DataFrame): dataframe with the column date added
    """
    data = data.sort_values('created_at')
    data['date'] = pd.to_datetime(data['created_at'], utc=True).dt.strftime('%Y-%m-%d')
    data['date'] = pd.to_datetime(data['date'])

    return data


def extract_hashtags(row, col: str = "text"):
    """
    Extracts all hashtags from the text
    """
    unique_hashtag_list = list(re.findall(r'#\S*\w', row[col]))
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
def word_freq(data: pd.DataFrame, stop_words: List[str], n_words:int=None):
    '''Calculating word frequency of tokenized tweets

    Args:
        data (pd.DataFrame): dataframe with the column tokens_string
        stop_words (List(str)): list of stopwords to exclude from frequency calculations
        n_words (int, optional): n_words top frequent words to return. Defaults to None, in which case all words are returned

    Returns: 
        df_freq (pd.DataFrame): dataframe with columns "word" and "frequency"
    '''
    w_freq = data.tokens_string.str.split(expand = True).stack().value_counts()
    w_freq = w_freq.to_frame().reset_index().rename(columns={'index': 'word', 0: 'Frequency'})
    w_freq = w_freq[-w_freq["word"].isin(stop_words)]

    if not n_words:
        df_freq= w_freq.nlargest(len(w_freq.index), columns=['Frequency'])
    else: 
        df_freq= w_freq.nlargest(n_words, columns=['Frequency'])
    return df_freq

# Bigrams
def get_bigrams(data: pd.DataFrame, 
                stop_words: List[str], 
                by_week=False,
                trouble_weeks ={'2020-12-28': '53-2020',
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
                                '2022-01-02': '52-2021'}):
    '''Calculating bigrams frequency on already tokenized words

    Args:
        data (pd.DataFrame): dataframe with columns date and tokens_list
        stop_words (List(str)): list of stopwords to exclude from frequency calculations
        by_week (bool, optional): whether or not to calculate bigrams on a weekly basis. Defaults to False
        trouble_weeks (dict, optional): dictionary to handle weeks around year change. Optional
    
    Returns
        bigram_df (pd.DataFrame): dataframe with bigrams and frequency, on a weekly basis if by_week == True
    '''
    bigram_df = pd.DataFrame()

    if by_week:
        data["Week_Y"] = [trouble_weeks.get(str(date)[:10], date.strftime('%V-%Y')) for date in data["date"]]
        weeks = list(data["Week_Y"].unique())
        for week in weeks:
            df = data[data["Week_Y"] == week]
            terms_bigram = []
            for tweet in df['tokens_list']:
                tokens = [token for token in ast.literal_eval(tweet) if token not in stop_words]
                terms_bigram.append(list(bigrams(tokens)))
            # Flatten list of bigrams in clean tweets
            bigrms = list(itertools.chain(*terms_bigram))
            bigram_counts = collections.Counter(bigrms)
            temp_df = pd.DataFrame(bigram_counts.most_common(30), columns=['bigram', 'count'])
            temp_df["week_y"] = week
            bigram_df = pd.concat([bigram_df, temp_df])
    
    else:
        terms_bigram = []
        for tweet in data['tokens_list']:
            tokens = [token for token in ast.literal_eval(tweet) if token not in stop_words]
            terms_bigram.append(list(bigrams(tokens)))
        # Flatten list of bigrams in clean tweets
        bigrms = list(itertools.chain(*terms_bigram))
        # Create counter of words in clean bigrams
        bigram_counts = collections.Counter(bigrms)
        bigram_df = pd.DataFrame(bigram_counts.most_common(30), columns=['bigram', 'count'])

        terms_bigram = []
        for tweet in data['tokens_list']:
            tokens = [token for token in ast.literal_eval(tweet) if token not in stop_words]
            terms_bigram.append(list(bigrams(tokens)))
        # Flatten list of bigrams in clean tweets
        bigrms = list(itertools.chain(*terms_bigram))
        
        # Create counter of words in clean bigrams
        bigram_counts = collections.Counter(bigrms)
        bigram_df = pd.DataFrame(bigram_counts.most_common(30), columns=['bigram', 'count'])

    return bigram_df

# wordcloud preparation
def wordcloud_prep(data: pd.DataFrame) -> pd.DataFrame:
    texts = data["tokens_string"]
    texts = texts[texts.notnull()] #Sometimes necessary
    texts = ", ".join(texts)

    return texts


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

def set_lab_freq(small: bool):
    '''
    Function for getting the correct column with smoothed values
    '''
    if small:
        return 's500_nr_of_tweets'
        
    else:
        return 's5000_nr_of_tweets'

def tweet_freq(small: bool):
    """Calculates the number of tweets per day.
    Returns pd.DataFrame with columns indicating date, nr of tweets that day and mean

    Args:
        keyword (str): the keyword indicating query term

    Returns:
        pd.DataFrame: the dataframe with the tweet frequencies
    """
    # Get name of column with smoothed frequencies
    name = set_lab_freq(small)

    # Calculate mean of day
    mean = df.groupby('date')[name].mean()
    mean = mean.reset_index()
    mean = mean.rename(columns = {name:'mean'})

    # Keep only necessary
    merged  = pd.merge(mean, df, on = 'date')
    merged = merged.filter(['date', 'nr_of_tweets', 'mean'], axis=1)

    return merged

def compute_ci(data: pd.DataFrame,
               array: str,
               upper_bound: str,
               lower_bound: str):
    '''
    Computes 95 percent confidence interval (CI)
    
    
    Args:
        data (str): The dataframe with the dataset
        array (str): The dataframe column with arrays of values
        upper_bound (str): The name of the output dataframe column with CI upper bound values
        lower_bound (str): The name of the output dataframe column with CI lower bound values
    
    
    Returns:
          pd.DataFrame: The input dataframe with two new columns (upper and lower bound values)
    '''
    # Prepare dataframe
    data[upper_bound] = 0.0
    data[lower_bound] = 0.0

    # Loop over rows in dataframe and add to 
    for i in range(len(data)):
        CI = sns.utils.ci(sns.algorithms.bootstrap(data[array][i])) 
        
        data.loc[i, upper_bound] = CI[0]
        data.loc[i, lower_bound] = CI[1]
        # data[upper_bound][i] = CI[0]
        # data[lower_bound][i] = CI[1]
        
    return data

def mean_ci_sentiment(df: pd.DataFrame):
    """Calculates mean and CI for the two smoothed sentiment columns (Vader and BERT)

    Args:
        df (pd.DataFrame): the dataframe with the smoothed sentiment columns

    Returns:
        pd.DataFrame: dataframe with the mean and CI columns added
    """
    # Filter only relevant columns    
    df0 = df.filter(['date', 'centered_compound', 'polarity_score_z'], axis=1)

    # Group compound sentiment scores by date. Make arrays from the grouped values
    df_compound = df0.groupby(['date'])['centered_compound'].apply(list).apply(lambda x: np.asarray(x))
    df_compound = df_compound.reset_index()
    df_compound = df_compound.rename(columns = {'centered_compound':'compound_array'})

    # Calculate mean for arrays of compound scores
    mean_compound = df0.groupby("date")['centered_compound'].mean()
    mean_compound  = mean_compound.reset_index()
    mean_compound = mean_compound.rename(columns = {'centered_compound':'compound_mean'})

    # Calculate upper and lower bounds for 95% confidence interval for compound score
    df_compound = compute_ci(df_compound, 
                            array = 'compound_array',
                            upper_bound = 'compound_upper',
                            lower_bound = 'compound_lower')

    # Merge mean and arrays for compund scores into a single df
    merged_compound = pd.merge(df_compound, mean_compound, on = 'date')

    # Group z sentiment scores by date. Make arrays from the grouped values
    df_z = df0.groupby(['date'])['polarity_score_z'].apply(list).apply(lambda x: np.asarray(x))
    df_z = df_z.reset_index()
    df_z = df_z.rename(columns = {'polarity_score_z':'z_array'})

    # Calculate mean for arrays of compound scores
    mean_z = df0.groupby("date")['polarity_score_z'].mean()
    mean_z  = mean_z.reset_index()
    mean_z = mean_z.rename(columns = {'polarity_score_z':'z_mean'})

    # Calculate upper and lower bounds for 95% confidence interval for z score
    df_z = compute_ci(df_z, 
                    array = 'z_array',
                    upper_bound = 'z_upper',
                    lower_bound = 'z_lower')

    # Merge mean and arrays for z scores into a single df
    merged_z= pd.merge(df_z, mean_z, on = 'date')

    # Create df with all values for z and compound scores 
    df_merged = pd.merge(merged_compound, merged_z, on = 'date')
    merged_final = pd.merge(df, df_merged, on = 'date')
    merged_final = merged_final.drop_duplicates(subset=['date'])

    return merged_final



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
            language = config[f'{key}']["lan"]
    
    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    test_limit = None if test_limit == 'None' else test_limit
    small = literal_eval(small)

    return keywords, from_date, to_date, small, language


if __name__ == "__main__":
    print("\n---------- Running streamlit_prep.py ----------\n")
    keywords, from_date, to_date, small, language = main(sys.argv[1:])
    ori_keyword_list = keywords.split(",")
    
    keyword_list = []
    for keyword in ori_keyword_list:
        if re.findall("~#", keyword):
            keyword = re.sub('~', '', keyword)
        else:
            keyword = re.sub("~", " ", keyword)
        keyword_list.append(keyword)

    data_prefix = keyword_list[0]

    # Check if a file with suffix _data exists
    # This means that there is new data to process
    # If not, just quit the pipeline for this query
    new_data = os.path.join("..", f'{data_prefix}_files', f'{data_prefix}_data.csv')
    if not os.path.exists(new_data):
        quit()

    print('--Running streamlit preparation--')
    print('Data prefix =', data_prefix, '\n---------------\n')
    ## load stopwords ##
    if language == 'da':
        sp = spacy.load('da_core_news_lg')
        file = open("stop_words.txt","r+")
        stop_words = file.read().split()
        stop_words = set(stop_words + ["'s"])
    if language == 'en':
        sp = spacy.load('en_core_web_lg')
        stop_words = set(STOPWORDS)
        stop_words.add("'s")
    
    # Tokenize and Lemmatize stop words
    tokenizer = sp.tokenizer
    joint_stops = " ".join(stop_words)
    tokenized = tokenizer(joint_stops).doc.text
    stops = sp(tokenized)
    lemma_stop_words = [t.lemma_ for t in stops]
    lemma_stop_words = list(set(lemma_stop_words))

    ## Load file ##
    df = load_data(data_prefix=data_prefix)

    ## make .pkl files ##
    out_path = os.path.join('..', f'data_{language}', f'{data_prefix}')
    # Make sure folders are ready
    if not os.path.exists(os.path.join('..', f'data_{language}')):
        os.makedirs(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ## Prepare df ##
    df = prepare_date_col(df)
    df = get_tweet_frequencies(df) 
    # save_file(df, out_path, f'{data_prefix}.pkl') # Not used??

    # Hashtags
    hashtags = hashtag_per_row(df)
    freq_hashtags = get_hashtag_frequencies(hashtags)
    df1 = freq_hashtags.sort_values(by=['nr_of_hashtags'], ascending=False)
    hashtags = df1.nlargest(30, columns=['nr_of_hashtags'])
    save_file(hashtags, out_path, f'{data_prefix}_hash.pkl')

    # Word frequency
    w_freqs = word_freq(df, lemma_stop_words+keyword_list, 30)
    save_file(w_freqs, out_path, f'{data_prefix}_w_freq.pkl')

    bi_w_freqs = word_freq(df, lemma_stop_words)
    save_file(bi_w_freqs, out_path, f'{data_prefix}_bigram_token_freq.pkl')

    # Bigrams
    bigrams_by_week = get_bigrams(df, lemma_stop_words, by_week=True) 
    bigrams_all_time = get_bigrams(df, lemma_stop_words) 
    save_file(bigrams_by_week, out_path, f'{data_prefix}_weekly_bigrams.pkl')
    save_file(bigrams_all_time, out_path, f'{data_prefix}_bigrams.pkl')

    # Wordcloud
    # texts = wordcloud_prep(df)
    # wordcloud_dict = {'texts': texts, 'my_stop_words': lemma_stop_words+keyword_list}
    # save_dict(wordcloud_dict, out_path, f'{data_prefix}_wordcloud.pkl') # the actual .png is used, not the .pkl file

    # Tweet frequency
    tweet_frequency = tweet_freq(small)
    save_file(tweet_frequency, out_path, f'{data_prefix}_tweet_freq.pkl')

    # Mean and CI sentiment
    sentiment_mean_ci = mean_ci_sentiment(df)
    save_file(sentiment_mean_ci, out_path, f'{data_prefix}_sentiment.pkl')
