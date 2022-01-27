import re
import sys
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
import os
import nltk
from nltk import bigrams

import spacy
from spacy.lang.da import Danish

from wordcloud import WordCloud

from pathlib import Path
from configparser import ConfigParser

################################################################################################
## PREPARE DATA FUNCTIONS
################################################################################################
activated = spacy.prefer_gpu()
sp = spacy.load('da_core_news_lg')
nlp = Danish()
tokenizer = nlp.tokenizer

file = open("stop_words.txt","r+")
stop_words = file.read().split()

# Tokenize and Lemmatize stop words
joint_stops = " ".join(stop_words)
tokenized = tokenizer(joint_stops).doc.text
stops = sp(tokenized)
my_stop_words = [t.lemma_ for t in stops]
my_stop_words = list(set(my_stop_words))

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

def join_tokens(row):
    res = " ".join(row["tokens_list"])
    return res

def lemmatize_tweet(tweet):
    tweet = clean_tweet(tweet)
    tokenized = tokenizer(tweet).doc.text
    spacy_tokens = sp(tokenized)
    lemmatized_tweet = [t.lemma_ for t in spacy_tokens]
    
    lemmatized_tweet = [x for x in lemmatized_tweet if x not in my_stop_words]
    
    hmm = ['   ','  ',' ','','♂','','❤','','🤷','”', '“', "'", '"', '’']
    lemmatized_tweet = [x for x in lemmatized_tweet if x not in hmm]
    lemmatized_tweet = [name for name in lemmatized_tweet if name.strip()]
    
    return lemmatized_tweet

def prep_word_freq(freq_df):
    freq_df["tokens_list"] = freq_df.mentioneless_text.apply(lemmatize_tweet)
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

################################################################################################
## DATA VISUALIZATION FUNCTIONS
################################################################################################

##### --- BASE PLOTTING SETTING --- #####
def set_base_plot_settings(fontsize, if_palette):
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    themes.theme_minimal(grid=False, ticks=False, fontsize=fontsize)
    a4_dims = (25,15)
    
    if if_palette:
        #          0 black      1 orange  2 L blue   3 green    4 L orange  5 D blue  6 D orange 7 purple
        palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    else:
        palette = 0
    
    fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)

    return fig, ax1, palette

def set_late_plot_settings(fig, ax1, if_dates):
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)

    ax1.grid(color='darkgrey', linestyle='-', linewidth=0.5, which= "both")
    if if_dates:
        # Define the date  format
        ax1.xaxis_date()
        date_form = mdates.DateFormatter("%d-%b")
        ax1.xaxis.set_major_formatter(date_form)

    ax1.set(ylim=(0, None))
    return fig, ax1

def set_late_barplot_settings(fig, ax1):
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)
    return fig, ax1

##### --- VISUALIZATION FUNCTIONS --- #####
def vis_keyword_mentions_freq(data_prefix, root_path, df, title, ysmooth_nr1, ysmooth_nr2):
    print("Visualize keyword mentions frequency")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)

    ax1 = sns.lineplot(x="date", y="nr_of_tweets", 
                      palette = palette[0], 
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y=ysmooth_nr1, 
                  color = palette[5], 
                     linewidth = 5, data = df)
        
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)

    plot_name = "../fig/" + f'{data_prefix}/' + data_prefix + ysmooth_nr1 + "_freq_mentions.png"
    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_{ysmooth_nr1}_freq_mentions.png')
    # plot_name = "../fig/" + data_prefix + ysmooth_nr1 + "_freq_mentions.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)

    ax1 = sns.lineplot(x="date", y="nr_of_tweets", 
                      palette = palette[0], 
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y=ysmooth_nr2, 
                  color = palette[5], 
                     linewidth = 5, data = df)
        
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)

    # plot_name = root_path + "fig/" + data_prefix + ysmooth_nr2 + "_freq_mentions.png"
    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_{ysmooth_nr2}_freq_mentions.png')
    fig.savefig(plot_name, bbox_inches='tight')
    print("Save figure done\n------------------\n")


def vis_hashtag_freq(data_prefix, root_path, df, nr_of_hashtags):
    print("Visualize hashtag frequency")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = False)
    
    df0 = df.nlargest(nr_of_hashtags, columns=['nr_of_hashtags'])
    nr_hash = len(df0["hashtag"].unique())
    palette = sns.color_palette("inferno", nr_hash)

    ax1 = sns.barplot(y="hashtag", x="nr_of_hashtags", palette = palette, data = df0)

    fig, ax1 = set_late_barplot_settings(fig, ax1)

    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_frequent_hashtags.png')
    # plot_name = root_path + "fig/" + data_prefix + "_frequent_hashtags.png"
    fig.savefig(plot_name, bbox_inches='tight')
    print("Save figure done\n------------------\n")
    
def vis_sentiment_compound(data_prefix, root_path, df, ysmooth_c1, ysmooth_c2):
    print("Visualize sentiment compound")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)

    ax1 = sns.lineplot(x="date", y="centered_compound", 
                       color = palette[2],
                         linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y=ysmooth_c1, 
                       color = palette[5],
                         linewidth = 5, data = df)

    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    ax1.set(ylim=(-1, 1))

    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}{ysmooth_c1}_sentiment_compound.png')
    # plot_name = root_path + "fig/" + data_prefix + ysmooth_c1 + "_sentiment_compound.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)

    ax1 = sns.lineplot(x="date", y="centered_compound", 
                       color = palette[2],
                         linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y=ysmooth_c2, 
                       color = palette[5],
                         linewidth = 5, data = df)

    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    ax1.set(ylim=(-1, 1))

    # plot_name = root_path + "fig/" + data_prefix + ysmooth_c2 + "_sentiment_compound.png"
    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}{ysmooth_c2}_sentiment_compound.png')
    fig.savefig(plot_name, bbox_inches='tight')
    print("Save figure done\n------------------\n")

def vis_word_freq(data_prefix, root_path, word_freq, nr_of_words):
    print("Visualize word frequency")
    
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = False)
    
    df0 = word_freq.nlargest(nr_of_words, columns=['Frequency'])
    nr_hash = len(df0["Word"].unique())

    palette = sns.color_palette("Blues_r", nr_hash)

    ax1 = sns.barplot(y="Word", x="Frequency", palette = palette, data = df0)

    fig, ax1 = set_late_barplot_settings(fig, ax1)

    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_word_frequency.png')
    fig.savefig(plot_name, bbox_inches='tight')
    print("Save figure done\n------------------\n")
    
def vis_word_cloud(data_prefix, root_path, wordcloud):
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_word_cloud.png')
    plt.savefig(plot_name, bbox_inches='tight')
    
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
    
def vis_bigram_graph(data_prefix, root_path, d, graph_layout_number):
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

    pos = nx.spring_layout(G, k=3)

    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=10,
                     width=3,
                     edge_color= "silver",
                     node_color= palette[2],
                     node_size = 1000,
                     with_labels = False,
                     ax=ax)

    # Create offset labels
    for key, value in pos.items():
        x, y = value[0], value[1]
        ax.text(x, y,
                s=key,
                #bbox=dict(facecolor= palette[2],
                #          alpha= 0.10),
                horizontalalignment='center', fontsize=14)


    fig.patch.set_visible(False)
    ax.axis('off')

    plot_name = os.path.join(root_path, "fig", f'{data_prefix}', f'{data_prefix}_bigram_graph_k{graph_layout_number}.png')
    fig.savefig(plot_name, dpi=150, bbox_inches='tight')
    print("Save figure done\n------------------\n")

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def visualize(data_prefix, root_path, ysmooth_nr1, ysmooth_nr2, ysmooth_c1, ysmooth_c2):
    # filename = f'{root_path}{data_prefix}_files/{data_prefix}_smoothed.csv'
    filename = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_smoothed.csv')
    df = pd.read_csv(filename,lineterminator='\n')
    
    # Create a column which is just date
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')

    #freq_df = get_tweet_frequencies(df)
    
    df["date"] = pd.to_datetime(df["date"])
    df['date_ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())

    # Visualize
    title = "Mentions of: " + str(data_prefix)
    vis_keyword_mentions_freq(data_prefix, root_path, df, title, ysmooth_nr1, ysmooth_nr2)
    
    freq_hashtags = get_hashtag_frequencies(df)
    hash_df = freq_hashtags.sort_values(by=['nr_of_hashtags'], ascending=False)
    vis_hashtag_freq(data_prefix, root_path, hash_df, nr_of_hashtags = 30)
    
    # Sentiment Analysis
    # Rolling average
    #freq_df['compound_7day_ave'] = df.compound.rolling(7).mean().shift(-3)
    if language == 'da':
        vis_sentiment_compound(data_prefix, root_path, df, ysmooth_c1, ysmooth_c2)
    
    ## WORD FREQUENCY
    print("Get word frequency")
    texts, word_freq = prep_word_freq(df)
    vis_word_freq(data_prefix, root_path, word_freq, nr_of_words = 30)
    
    # WORD CLOUD
    #%matplotlib inline
    if language == 'da':
        file = open("stop_words.txt","r+")
        stop_words = file.read().split()
        # Generate word cloud
        wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                            background_color='white', colormap="rocket", 
                            collocations=False, stopwords = stop_words).generate(texts)

        vis_word_cloud(data_prefix, root_path, wordcloud)
    
    # BIGRAM GRAPH
    d = create_bigrams(df)
    k_numbers_to_try = [1,2,3,4,5]
    for k in k_numbers_to_try:
        vis_bigram_graph(data_prefix, root_path, d, graph_layout_number = k)
    
    
    print(df.head())
    print(df.columns)
    # output_name = root_path + data_prefix + "_final.csv"
    # output_name = f'{root_path}{data_prefix}_files/{data_prefix}_final.csv'
    output_name = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_final.csv')
    df.to_csv(output_name,index = False)
    
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
            print(f'Running visualizations with key: {key}, keywords: {keywords} from {from_date}. Small = {small}. Language = {language}.')
    
    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    test_limit = None if test_limit == 'None' else test_limit

    return keywords, from_date, to_date, small, language

########################################################################################################################
##     INPUT
########################################################################################################################

if __name__ == "__main__":
    
    keywords, from_date, to_date, small, language = main(sys.argv[1:])
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
    
    if small == "True":
        if_small = True
    elif small == "False":
        if_small = False
    
    ## conditional for ysmooth depending on small
    if if_small:
        ysmooth_nr1 = "s200_nr_of_tweets"
        ysmooth_nr2 = "s500_nr_of_tweets"
        ysmooth_c1 = "s200_compound"
        ysmooth_c2 = "s500_compound"
    else:
        ysmooth_nr1 = "s2000_nr_of_tweets"
        ysmooth_nr2 = "s5000_nr_of_tweets"
        ysmooth_c1 = "s2000_compound"
        ysmooth_c2 = "s5000_compound" 
    
    ###############################
    print("---VISUALIZE---")
    print("START loading data: ", data_prefix)
    
    #Path("/fig").mkdir(parents=True, exist_ok=True)
    visualize(data_prefix, root_path, ysmooth_nr1, ysmooth_nr2, ysmooth_c1, ysmooth_c2)