"""
Extract the data with your specified keywords and filtering from the collected Twitter corpus, specify date range if relevant

"""

import pandas as pd
import glob
import ndjson
import re
import string
import getopt, sys
import os
import os.path
from os import path
from icecream import ic

def retrieve_retweets(row):
    if re.match("^RT", row):
        RT = True
    else:
        RT = False
    return RT

def remove_retweets(data):
    patternDel = "^RT"
    data["text"] = data["text"].astype(str)
    filtering = data['text'].str.contains(patternDel)
    removed_RT = data[~filtering].reset_index(drop=True)
    return removed_RT

def extract_usernames(row):
    username_list = list(re.findall(r'@(\S*)\w', row["text"]))
    return username_list

def extract_keywords(row, keyword_list):
    if type(row["text"]) != str:
        print("Broken text: ", row["text"])
        row["text"] = row["text"].astype(str)
    tweet = row["text"].lower()
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    res = [ele for ele in keyword_list if(ele in tweet)] 
    return res

def remove_date_dash(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_processed_files_from_path(output_name):
    ori_df = pd.read_csv(output_name)
    ori_df = ori_df[ori_df["0"] != 'created_at'].reset_index(drop=True)
        
    dates = pd.to_datetime(ori_df["0"].dropna()[1:], utc=True).dt.strftime('%Y-%m-%d').drop_duplicates().reset_index(drop = True).astype(str)
    dates_to_ignore = dates.apply(remove_date_dash).to_list()
        
    files_to_ignore = []
    for file in mega_path:
        date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
        if date in dates_to_ignore:
            files_to_ignore.append(file)

    for element in files_to_ignore:
        if element in mega_path:
            mega_path.remove(element)
            
    return mega_path

def remove_dates_lessthan_from_date(from_date_str):
    files_to_ignore = []
    for file in mega_path:
        date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
        if date < from_date_str:
            files_to_ignore.append(file)

    for element in files_to_ignore:
        if element in mega_path:
            mega_path.remove(element)
            
    return mega_path

def ignore_dates_less_than(output_name):
    ori_df = pd.read_csv(output_name)
    ori_df = ori_df[ori_df["0"] != 'created_at'].reset_index(drop=True)
        
    dates = pd.to_datetime(ori_df["0"].dropna()[1:], utc=True).dt.strftime('%Y-%m-%d').drop_duplicates().reset_index(drop = True).astype(str)
    
    # Sort the dates
    
    dates_to_ignore = dates.apply(remove_date_dash).to_list()
    
    maximum_date = remove_date_dash(max(dates))
        
    files_to_ignore = []
    for file in mega_path:
        date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
        if date <= maximum_date:
            files_to_ignore.append(file)

    for element in files_to_ignore:
        if element in mega_path:
            mega_path.remove(element)
            
    return mega_path

def extract_data(keyword_list, data_prefix, mega_path, from_date):
    print("START data extraction for keywords: ", keyword_list)
    print("---")
    
    output_name = "../" + data_prefix + "_data.csv"
    print("Does the file already exist?: ", path.exists(output_name))
    
    if path.exists(output_name):
        mega_path = ignore_dates_less_than(output_name)
    print("Go through files: \n")    
    mega_path.sort()
    
    ic(mega_path)
    
    # Create a directory for these files
    temp_path = "../tmp_" + data_prefix + "/"

    try:
        os.mkdir(temp_path)
    except OSError:
        print ("Creation of the directory %s failed" % temp_path)
    else:
        print ("Successfully created the directory %s " % temp_path)
        
    for file in mega_path:
        file_name = re.findall(r'(td.*)\.ndjson', file)[0]

        print("Opening          " +  file_name)

        data = pd.read_json(file, lines = True)
        print("Begin processing " +  file_name)

        df = data[["created_at", "id", "text"]]
        df = df.dropna()

        df = remove_retweets(df)
        df["search_keyword"] = df.apply(lambda row: extract_keywords(row, keyword_list), axis = 1)
        
        df["search_keyword"] = df["search_keyword"].astype(str)
        df = df[df["search_keyword"] != "[]"].drop_duplicates().reset_index(drop=True)
                
        if len(df) > 0:
            filename = temp_path + "/" + data_prefix + "_" + file_name + ".csv"
            df.to_csv(filename, index = False)

            print("Save of " + file_name + " done")
        else:
            print("Not enough data")
        print("-------------------------------------------\n")
        del df

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

    if test_limit:
        pathname = '/data/001_twitter_hope/preprocessed/da/td_' + str(test_limit) + '*.ndjson'
        ic(pathname)
        mega_path = glob.glob(pathname)
        ic(mega_path)
    else:
        # Runs through all the files here, date does not matter
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        mega_path = glob.glob(pathname)

    ###############################
    print("--------EXTRACT DATA--------")
    extract_data(keyword_list, data_prefix, mega_path, from_date)