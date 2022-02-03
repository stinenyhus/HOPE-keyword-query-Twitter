"""
Get the smoothing going and entropy
"""

import pandas as pd
import numpy as np
from icecream import ic
import datetime
import matplotlib.pyplot as plt
import datetime as dt
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler
from configparser import ConfigParser
from ast import literal_eval

import re
import string
import getopt, sys
import os
import os.path
from os import path

########################################################################################################################
##     DEFINE FUNCTIONS
########################################################################################################################

def entropy2(labels, 
             base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent

def gaussian_kernel(arr, 
                    sigma=False, 
                    fwhm=False):
    """ gaussian kernel smoother for signal arr
    - sigma: standard deviation of gaussian distribution
    - fwhm: full width at half maximum of gaussian distribution
    """
    y_vals = np.array(arr)
    x_vals = np.arange(arr.shape[0])
    if sigma == fwhm:
        print("[INFO] Define parameters \u03C3 xor FWHM")
        pass
    elif fwhm:
        sigma = fwhm / np.sqrt(8 * np.log(2))
    else:
        sigma = sigma
        fwhm = sigma * np.sqrt(8 * np.log(2))
    print("[INFO] Applying Gaussian kernel for \u03C3 = {} and FWHM = {} ".format(round(sigma,2), round(fwhm,2)))
    smoothed_vals = np.zeros(y_vals.shape)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_vals * kernel)
    return smoothed_vals

def get_tweet_frequencies(df):
    tweet_freq = pd.DataFrame({'nr_of_tweets' : df.groupby(['date']).size()}).reset_index()
    freq_tweets = pd.merge(df, tweet_freq, how='left', on=['date'])
    return freq_tweets

def apply_date_mask(df, from_date):
    mask = (df['date'] >= from_date)
    df = df.loc[mask]
    return df

def center_compound(df):
    print(df.head())
    print("Sum compound ", sum(df["compound"]))
    print("Len compound ", len(df["compound"]))
    
    average_compound = sum(df["compound"]) / len(df["compound"])
    df["centered_compound"] = df["compound"] - average_compound
    return df

def get_entropy(df):
    entropy_df = df.set_index(pd.to_datetime(df['created_at'], utc = True))
    entropy_df = entropy_df.sort_values("date")

    entropy = []
    day = []
    for u,v in entropy_df.groupby(pd.Grouper(freq="D")):
        ent = entropy2(v['centered_compound'])
        entropy.append(ent)
        day.append(u)

    entropy_df = pd.DataFrame(entropy, day).reset_index().rename(columns={"index":"day", 0:"entropy"})
    entropy_df["day"] = pd.to_datetime(entropy_df["day"], utc=True).dt.strftime('%Y-%m-%d')
    df["day"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')

    df = pd.merge(df, entropy_df, on= "day")
    return df

def center_entropy(df):
    min_entropy = min(df["entropy"])
    max_entropy = max(df["entropy"])
    difference = max_entropy - min_entropy

    df["normalized_entropy"] = (df["entropy"] - min_entropy) / difference
    average_entropy = sum(df["normalized_entropy"]) / len(df["normalized_entropy"])
    df["centered_entropy"] = df["normalized_entropy"] - average_entropy
    return df

def smooth_2000(df, if_compound, if_nroftweets, if_bert, if_small):
    if if_compound:
        if if_small:
            print("Compound FWHM = 200")
            df["s200_compound"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 200)
        else:
            print("Compound FWHM = 2000")
            df["s2000_compound"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 2000)
    if if_nroftweets:
        if if_small:
            print("Nr FWHM = 200")
            df["s200_nr_of_tweets"] = gaussian_kernel(df["nr_of_tweets"], sigma = 1, fwhm = 200)
        else:
            print("Nr FWHM = 2000")
            df["s2000_nr_of_tweets"] = gaussian_kernel(df["nr_of_tweets"], sigma = 1, fwhm = 2000)
    if if_bert:
        if if_small:
            print("BERT FWHM = 200")
            df["s200_polarity_score_z"] = gaussian_kernel(df["polarity_score_z"], sigma = 1, fwhm = 200)
        else:
            print("BERT FWHM = 2000")
            df["s2000_polarity_score_z"] = gaussian_kernel(df["polarity_score_z"], sigma = 1, fwhm = 2000)
    return df

def smooth_5000(df, if_compound, if_nroftweets, if_bert, if_small):
    if if_compound:
        if if_small:
            print("Compound FWHM = 500")
            df["s500_compound"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 500)
        else:
            print("Compound FWHM = 5000")
            df["s5000_compound"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 5000)
    if if_nroftweets:
        if if_small:
            print("Nr FWHM = 500")
            df["s500_nr_of_tweets"] = gaussian_kernel(df["nr_of_tweets"], sigma = 1, fwhm = 500)
        else:
            print("Nr FWHM = 5000")
            df["s5000_nr_of_tweets"] = gaussian_kernel(df["nr_of_tweets"], sigma = 1, fwhm = 5000)
    if if_bert:
        if if_small:
            print("BERT FWHM = 500")
            df["s500_polarity_score_z"] = gaussian_kernel(df["polarity_score_z"], sigma = 1, fwhm = 500)
        else:
            print("BERT FWHM = 5000")
            df["s5000_polarity_score_z"] = gaussian_kernel(df["polarity_score_z"], sigma = 1, fwhm = 5000)
    return df

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def smooth_and_entropy(data_prefix: str, 
                       root_path: str, 
                       from_date: str, 
                       if_compound: bool, 
                       if_nroftweets: bool, 
                       if_bert: bool,
                       if_entropy: bool, 
                       if_small=True):
    """Main smoothing function, the processes that occur depend on the booleans
    data_prefix: str, 
    root_path: str, 
    from_date: str, 
    if_compound: bool, smooth sentiment compound 
    if_nroftweets: bool, smooth number of tweets
    if_entropy: bool, calculate and smooth entropy
    if_small=True
    """
    print("Read in data, prepare")
    # vis_file = root_path + data_prefix + "_vis.csv"
    vis_file = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_vis.csv')
    df = pd.read_csv(vis_file,lineterminator='\n')
    df = df.sort_values("created_at")
    print(len(df))
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')
    df["date"] = pd.to_datetime(df["date"])
    
    if from_date:
        df = apply_date_mask(df, from_date)
    if len(df) < 1:
        print("ERROR: Apply a from_date in the bash command with e.g. '-f 2021-01-01'.")
    
    print("Check for compound")
    if 'compound' in df.columns:
        print("Center compound and entropy")
    else:
        print("No compound")
        ic(df.head())
        ic(df.columns)
    
    df = center_compound(df)
    df = get_entropy(df)
    df = center_entropy(df)

    if if_bert:
        X = np.array(df["polarity_score"]).reshape(-1, 1)
        df['polarity_score_z'] = StandardScaler().fit_transform(X)
    
    print("START SMOOTHING")
    df = smooth_2000(df, if_compound, if_nroftweets, if_bert, if_small)
    print("Smooth1 DONE")
    df = smooth_5000(df, if_compound, if_nroftweets, if_bert, if_small)
    print("Smooth2 DONE")

    comment = []
    if if_compound:
        comment.append("_compound")
    if if_nroftweets:
        comment.append("_nroftweets")
    
    # outfile_name = root_path + data_prefix + "_smoothed.csv"
    outfile_name = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_smoothed.csv')
    df.to_csv(outfile_name, index=False)
    del df

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
            print(f'Running smoothing pipeline with key: {key}, keywords: {keywords} from {from_date}. Small = {small}. Language = {language}.')
    
    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    test_limit = None if test_limit == 'None' else test_limit
    small = literal_eval(small)

    return keywords, test_limit, from_date, small, language

########################################################################################################################
##     INPUT
########################################################################################################################

if __name__ == "__main__":
    
    keywords, test_limit, from_date, small, language = main(sys.argv[1:])
    ic(main(sys.argv[1:]))
    ori_keyword_list = keywords.split(",")
    
    keyword_list = []
    for keyword in ori_keyword_list:
        if re.findall("~#", keyword):
            keyword = re.sub('~', '', keyword)
        else:
            keyword = re.sub("~", " ", keyword)
        keyword_list.append(keyword)
    
    ic(keyword_list)

    data_prefix = keyword_list[0]
    # root_path = "/home/commando/stine-sara/HOPE-keyword-query-Twitter/"
    root_path = os.path.join("..") 

    ###############################
    print("--------SMOOTHING PIPELINE START--------")

    smooth_and_entropy(data_prefix, root_path, from_date, 
                       if_compound = True, if_nroftweets = True, 
                       if_bert=True, if_entropy=False, if_small = small)