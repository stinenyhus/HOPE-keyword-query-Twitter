import re
import sys
import spacy
import getopt
import warnings
import itertools
import collections

import pandas as pd
import networkx as nx
import seaborn as sns; sns.set()
import pyplot_themes as themes
import datetime as dt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import string
from string import digits

import nltk
from nltk import bigrams

from wordcloud import WordCloud


sp = spacy.load('da_core_news_lg')

file = open("stop_words.txt","r+")
stop_words = file.read().split()

# Lemmatize stop words
stops = " ".join(stop_words)
stops = sp(stops)
my_stop_words = [t.lemma_ for t in stops]

def extract_hashtags(row):
    unique_hashtag_list = list(re.findall(r'#\S*\w', row["text"]))
    return unique_hashtag_list

def hashtag_per_row(data):
    # Create hashtags column with the actual unique hashtags
    data["hashtags"] = data.apply(lambda row: extract_hashtags(row), axis = 1)

    # Let's take a subset of necessary columns, add id
    df = data[["date", "hashtags"]].reset_index().rename(columns={"index": "id"})

    # Select only the ones where we have more than 1 hashtag per tweet
    df = df[df["hashtags"].map(len) > 1].reset_index(drop=True)

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
    df["hashtag"] = df["hashtag"].str.lower()
    df["hashtag"] = df["hashtag"].str.replace("'.", "")
    df["hashtag"] = df["hashtag"].str.replace("’.", "")

    return df

def lemmas(row):
    tweet = row["mentioneless_text"].lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.replace('”', '')
    tweet = tweet.replace('“', '')
    tweet = tweet.replace('»', '')
    tweet = tweet.replace('…','')
    
    sentence = sp(tweet)
    
    lemmas = []
    for word in sentence:
        lemmas.append(word.lemma_)
    
    res = [x for x in lemmas if x not in my_stop_words]
    hmm = ['    ','   ','  ',' ','', '🇩','🇰', '♂', '🤷']
    res = [x for x in res if x not in hmm]
    return res

def join_tokens(row):
    res = " ".join(row["tokens_list"])
    return res

def prep_word_freq(freq_df):
    freq_df["tokens_list"] = freq_df.apply(lambda row: lemmas(row), axis = 1)
    freq_df["tokens_string"] = freq_df.apply(lambda row: join_tokens(row), axis = 1)
    texts = freq_df["tokens_string"]
    texts = ", ".join(texts)
    
    word_freq = freq_df.tokens_string.str.split(expand=True).stack().value_counts()
    word_freq = word_freq.to_frame().reset_index().rename(columns={"index": "Word", 0: "Frequency"})
    
    return texts, word_freq

# Aggregate a frequency DF
def get_hashtag_frequencies(df):
    df = hashtag_per_row(df)
    # Add freq of hashtags by themselves in the dataset
    tweet_freq = pd.DataFrame({'nr_of_hashtags' : df.groupby(['hashtag']).size()}).reset_index()
    
    return tweet_freq

def vis_keyword_mentions_freq(data_prefix, freq_df, title):
    print("Visualize keyword mentions frequency")
    
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rc('xtick', labelsize=20)
    themes.theme_minimal(grid=False, ticks=False, fontsize=18)
    a4_dims = (25,15) #(11.7, 8.27)
    fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)
    
    # Color blind friendly colors
    palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    ax1 = sns.lineplot(x="date", y="nr_of_tweets", 
                      palette = palette[0], 
                        linewidth = 3, data = freq_df)
    """
    ax1 = sns.regplot(
        data=freq_df,
        x='date_ordinal',
        y='nr_of_tweets',
        color = palette[3],
        ci=False
        #y_jitter=.5
    )
    """
    
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)
    ax1.grid(color='grey', linestyle='-', linewidth=0.5, which= "both")

    """
    plt.axvline(dt.datetime(2020, 12, 21), color=palette[5])
    plt.text(x = dt.datetime(2020, 12, 22), # x-coordinate position of data label, adjusted to be 3 right of the data point
     y = 81, # y-coordinate position of data label, to take max height 
     s = 'Flight ban from UK', # data label
     color = palette[5])

    plt.axvline(dt.datetime(2020, 12, 23), color=palette[5])
    plt.text(x = dt.datetime(2020, 12, 24), # x-coordinate position of data label, adjusted to be 3 right of the data point
     y = 22, # y-coordinate position of data label, to take max height 
     s = 'Flight ban from UK extended', # data label
     color = palette[5])

    plt.axvline(dt.datetime(2021, 1, 5), color=palette[5])
    plt.text(x = dt.datetime(2021, 1, 6), # x-coordinate position of data label, adjusted to be 3 right of the data point
     y = 142, # y-coordinate position of data label, to take max height 
     s = 'High alert, new 5m distance rules', # data label
     color = palette[5])

    plt.axvline(dt.datetime(2021, 1, 13), color=palette[5])
    plt.text(x = dt.datetime(2021, 1, 14), # x-coordinate position of data label, adjusted to be 3 right of the data point
     y = 110, # y-coordinate position of data label, to take max height 
     s = 'Lockdown extended', # data label
     color = palette[5])
    """
    # Define the date format
    ax1.xaxis_date()
    date_form = mdates.DateFormatter("%d-%m")
    ax1.xaxis.set_major_formatter(date_form)

    fig.suptitle(title, size = "40")
    plot_name = "../fig/" + data_prefix + "_freq_mentions.png"
    fig.savefig(plot_name)
    print("Save figure done\n------------------\n")

def vis_hashtag_freq(data_prefix, df, nr_of_hashtags):
    print("Visualize hashtag frequency")
    themes.theme_minimal(grid=False, ticks=False, fontsize=18)
    a4_dims = (25,15)
    
    fig, (ax) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)
    
    df0 = df.nlargest(nr_of_hashtags, columns=['nr_of_hashtags'])
    nr_hash = len(df0["hashtag"].unique())

    themes.theme_minimal(grid=False, ticks=False, fontsize=18)
    palette = sns.color_palette("inferno", nr_hash)

    ax = sns.barplot(y="hashtag", x="nr_of_hashtags", palette = palette, data = df0)

    ax.set(xlabel="Count", ylabel = "Hashtag")
    ax.xaxis.get_label().set_fontsize(25)
    ax.yaxis.get_label().set_fontsize(25)
    ax.axes.set_title("Most frequent hashtags",fontsize=30)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)

    plot_name = "../fig/" + data_prefix + "_frequent_hashtags.png"
    fig.savefig(plot_name)
    print("Save figure done\n------------------\n")
    
def vis_sentiment_compound(data_prefix, clean_df):
    print("Visualize sentiment compound")
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rc('xtick', labelsize=20)
    themes.theme_minimal(grid=False, ticks=False, fontsize=18)
    a4_dims = (25,15) #(11.7, 8.27)
    palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)
    ax1 = sns.lineplot(x="date", y="compound", 
                       label="Daily", color = palette[2],
                         linewidth = 3, data = clean_df)

    ax1 = sns.lineplot(x="date", y="compound_7day_ave", 
                       label="7 Day Average", color = palette[5],
                         linewidth = 5, data = clean_df)

    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)

    ax1.grid(color='grey', linestyle='-', linewidth=0.5, which= "both")

    # Define the date format
    ax1.xaxis_date()
    date_form = mdates.DateFormatter("%d-%m")
    ax1.xaxis.set_major_formatter(date_form)

    figtitle = "Sentiment analysis: Compound scores of " + data_prefix
    fig.suptitle(figtitle, size = "40")

    ax1.set(ylim=(-1, 1))

    plot_name = "../fig/" + data_prefix + "_sentiment_compound.png"
    fig.savefig(plot_name)
    print("Save figure done\n------------------\n")

def vis_word_freq(data_prefix, word_freq, nr_of_words):
    print("Visualize word frequency")
    a4_dims = (25,15)
    fig, (ax) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)
    
    df0 = word_freq.nlargest(nr_of_words, columns=['Frequency'])
    nr_hash = len(df0["Word"].unique())

    themes.theme_minimal(grid=False, ticks=False, fontsize=18)
    palette = sns.color_palette("Blues_r", nr_hash)

    ax = sns.barplot(y="Word", x="Frequency", palette = palette, data = df0)

    ax.set(xlabel="Count", ylabel = "Word")
    ax.xaxis.get_label().set_fontsize(25)
    ax.yaxis.get_label().set_fontsize(25)
    ax.axes.set_title("Most frequent words",fontsize=30)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)

    plot_name = "../fig/" + data_prefix + "_word_frequency.png"
    fig.savefig(plot_name)
    print("Save figure done\n------------------\n")
    
def vis_word_cloud(data_prefix, wordcloud):
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    plot_name = "../fig/" + data_prefix + "_word_cloud.png"
    plt.savefig(plot_name)
    
# Aggregate a frequency DF
def get_tweet_frequencies(df):
    # Add freq of hashtags by themselves in the dataset
    tweet_freq = pd.DataFrame({'nr_of_tweets' : df.groupby(['date']).size()}).reset_index()

    # Add the whole_frew to id_hashtag
    freq_hashtags = pd.merge(df, tweet_freq, how='left', on=['date'])#, 'id', 'created_at'])
    
    df0 = freq_hashtags
    return df0

def create_bigrams(freq_df):
    from nltk import bigrams
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Create list of lists containing bigrams in tweets
    terms_bigram = [list(bigrams(tweet)) for tweet in freq_df['tokens_list']]
    # Flatten list of bigrams in clean tweets
    bigrams = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams)
    bigram_df = pd.DataFrame(bigram_counts.most_common(30), columns=["bigram", "count"])
    
    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index("bigram").T.to_dict("records")
    return d
    
def vis_bigram_graph(data_prefix, d, graph_layout_number):
    print("Visualize bigrams")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    
    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 5))

    fig, ax = plt.subplots(figsize=(11, 9))

    pos = nx.spring_layout(G, k=graph_layout_number)

    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=10,
                     width=3,
                     edge_color= palette[0], #'red',
                     node_color= palette[2], #'green',
                     with_labels = False,
                     ax=ax)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.065
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor= palette[7], #'red', 
                          alpha=0.3), ## 0.5),
                horizontalalignment='center', fontsize=14)


    fig.patch.set_visible(False)
    ax.axis('off')
    
    plot_name = "../fig/" + str(data_prefix) + "_bigram_graph_k" + str(graph_layout_number) + ".png"
    fig.savefig(plot_name, dpi=150)
    print("Save figure done\n------------------\n")
    
def main(argv):
    keywords = ''
    from_date = '' 
    to_date = ''
    test_limit = '' # this is so that it's possible to test the system on just one day/month of data
    try:
        opts, args = getopt.getopt(argv,"hk:f:t:l:")
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
    print('Input keywords are ', keywords)
    return keywords, from_date, to_date#, test_limit - this is not necessary to output for preprocess_stats.py

def visualize(data_prefix):
    filename = "../" + data_prefix + "_vis.csv"
    df = pd.read_csv(filename)
    
    # Create a column which is just date
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')

    freq_df = get_tweet_frequencies(df)
    
    freq_df["date"] = pd.to_datetime(freq_df["date"])
    freq_df['date_ordinal'] = pd.to_datetime(freq_df['date']).apply(lambda date: date.toordinal())

    # Visualize
    title = "Mentions of: " + str(data_prefix)
    vis_keyword_mentions_freq(data_prefix, freq_df, title)
    
    freq_hashtags = get_hashtag_frequencies(freq_df)
    hash_df = freq_hashtags.sort_values(by=['nr_of_hashtags'], ascending=False)
    vis_hashtag_freq(data_prefix, hash_df, nr_of_hashtags = 30)
    
    # Sentiment Analysis
    # Rolling average
    freq_df["date"] = pd.to_datetime(df["date"])
    freq_df['compound_7day_ave'] = df.compound.rolling(7).mean().shift(-3)
    vis_sentiment_compound(data_prefix, freq_df)
    
    ## WORD FREQUENCY
    print("Get word frequency")
    texts, word_freq = prep_word_freq(freq_df)
    vis_word_freq(data_prefix, word_freq, nr_of_words = 30)
    
    # WORD CLOUD
    #%matplotlib inline
    file = open("stop_words.txt","r+")
    stop_words = file.read().split()
    # Generate word cloud
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                          background_color='white', colormap="rocket", 
                          collocations=False, stopwords = stop_words).generate(texts)

    vis_word_cloud(data_prefix, wordcloud)
    
    # BIGRAM GRAPH
    d = create_bigrams(freq_df)
    k_numbers_to_try = [1,2,3,4,5]
    for k in k_numbers_to_try:
        vis_bigram_graph(data_prefix, d, graph_layout_number = k)
    

if __name__ == "__main__":
    
    keywords, from_date, to_date = main(sys.argv[1:])
    keyword_list = keywords.split(",")

    data_prefix = keyword_list[0]
    
    ###############################
    print("---VISUALIZE---")
    print("START loading data: ", data_prefix)
    
    visualize(data_prefix)