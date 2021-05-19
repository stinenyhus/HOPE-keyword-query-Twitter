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

import re
import string
import getopt, sys
import os
import os.path
from os import path

def entropy2(labels, base=None):
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

def gaussian_kernel(arr, sigma=False, fwhm=False):
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

def apply_date_mask(df, from_date):
    mask = (df['date'] >= from_date)
    df = df.loc[mask]
    return df

def center_compound(df):
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

def smooth_2000(df):
    df["smooth_entropy"] = gaussian_kernel(df["entropy"], sigma = 1, fwhm = 2000)
    df["smooth_compound"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 2000)
    return df

def smooth_5000(df):
    df["smooth_entropy2"] = gaussian_kernel(df["entropy"], sigma = 1, fwhm = 5000)
    df["smooth_compound2"] = gaussian_kernel(df["centered_compound"], sigma = 1, fwhm = 5000)
    return df

###########################################################

def smooth_and_entropy(data_prefix, vis_file, from_date):
    print("Read in data, prepare")
    df = pd.read_csv(vis_file)
    df = df.sort_values("created_at")
    print(len(df))
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y-%m-%d')
    df["date"] = pd.to_datetime(df["date"])
    
    df = apply_date_mask(df, from_date)
    print("Center compound and entropy")
    df = center_compound(df)
    df = get_entropy(df)
    df = center_entropy(df)
    
    print("START SMOOTHING")
    df = smooth_2000(df)
    print("Smooth 2000 DONE")
    df = smooth_5000(df)
    print("Smooth 5000 DONE")
    
    outfile_name = "../" + data_prefix + "_smoothed.csv"
    df.to_csv(outfile_name, index=False)
    
    del df

###########################################################

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
    return keywords, test_limit, from_date#, to_date - these are not necessary to output for extract_data.py

if __name__ == "__main__":
    
    keywords, test_limit, from_date = main(sys.argv[1:])
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

    vis_file = "../" + data_prefix + "_vis.csv"

    ###############################
    print("--------SMOOTHING PIPELINE START--------")
    smooth_and_entropy(data_prefix, vis_file, from_date)