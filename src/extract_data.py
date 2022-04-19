"""
Extract the data with your specified keywords and filtering from the collected Twitter corpus, specify date range if relevant

"""

import pandas as pd
import glob
import re
import string
import getopt, sys
import os
import os.path
from os import path
from icecream import ic
from datetime import date
from configparser import ConfigParser
from ast import literal_eval
from typing import Iterable, List, Optional

from en.extract_en_data import extract_en_data

########################################################################################################################
##     DEFINE FUNCTIONS
########################################################################################################################

def remove_retweets(data: pd.DataFrame,
                    col: Optional[str] = "text"):
    """Finds tweets that are RTs (retweets) and removes them
    Args:
        data (pd.DataFrame):  Dataframe with (at least) column "text"
        col (optional, str): String specifying the column of interest
    
    Returns:
        removed_RT (pd.DataFrame): The input dataframe with the RTs removed
    """
    patternDel = "^RT"
    data["text"] = data[col].astype(str)
    filtering = data[col].str.contains(patternDel)
    removed_RT = data[~filtering].reset_index(drop=True)
    return removed_RT

def extract_keywords(row: Iterable, 
                     keyword_list: List[str],
                     col: Optional[str] = "text"):
    """Lowercases tweets, finds keyword matches. Function intended for mapping/applying
    Args:
        row (pd.Series): pandas DataFrame row
        keyword_list (list): list of keywords (str)
        col (optional, str): String specifying the column of interest
    
    Returns:
        res (list): the list of the extracted keywords
    """
    tweet = row[col].lower()
    res = [ele for ele in keyword_list if(ele in tweet)] 
    return res

def remove_date_dash(text: str):
    """Removes dash in dates when ignoring already processed files based on whether the date exists or is new
    Note: string is a module

    Args:
        text (str): date as a string
    
    Returns:
        text (str): the string modified, all punctuation is removed
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def ignore_dates_less_than(output_name: str, mega_path: List[str]):
    """Finds out which dates already exist in the processed dataset 
    Removes files from mega path that are less than the max date
    Assumes the original df has a column called created_at

    Args:
        output_name (str): path to the already existing output dataframe
        mega_path (List[str]): list of strings to all the files in the raw data

    Returns:
        mega_path (list): Modified list of only the files that should be processed
    """
    # Get the dataframe of already processed files 
    ori_df = pd.read_csv(output_name, lineterminator="\n")
    print(ori_df.head())

    # Turn created_at column into datetime format and get the maximum date
    dates = pd.to_datetime(ori_df["created_at"], utc=True).dt.strftime('%Y-%m-%d').drop_duplicates().reset_index(drop = True).astype(str)
    maximum_date = remove_date_dash(max(dates))

    # Looping over files to identidy already processed files
    files_to_ignore = []
    for file in mega_path:

        date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
        if date <= maximum_date:
            files_to_ignore.append(file)

    # Removing already processed files from path
    for element in files_to_ignore:
        if element in mega_path:
            mega_path.remove(element)
    
    print(f"Ignoring {len(files_to_ignore)} files and going through {len(mega_path)} files based on files that have already been processed")

    return mega_path


def define_megapath(test_limit: Optional[str] = None, from_date:  Optional[str] = None, to_date:  Optional[str] = None):
    """This function defines the mega_path containing all relevant files as specified by the limit, from- and to-dates

    Args:
        test_limit (optional, str): Only run pipeline on one file if test_limit is specified
        from_date (optional, str): Minimum date for tweets
        to_date (optional, str): Maximum date for tweets

    Returns:
        mega_path (List[str]): A list of filenames of the files that match the data specifications
    """
    # Only run on one file
    if test_limit:
        pathname = '/data/001_twitter_hope/preprocessed/da/td_' + str(test_limit) + '*.ndjson'
        ic(pathname)
        mega_path = glob.glob(pathname)
        ic(mega_path)
    
    # If both a from_date and to_date is specified, use files inbetween the two
    elif from_date and to_date:
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        
        min_date = remove_date_dash(from_date)
        max_date = remove_date_dash(to_date)
        mega_path = glob.glob(pathname)
        
        files_to_ignore = []
        for file in mega_path:
            date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
            if (date < min_date) or (date > max_date):
                files_to_ignore.append(file)
        
        
        for element in files_to_ignore:
            if element in mega_path:
                mega_path.remove(element)
    
    # If only from_date is provided, use all dates after that
    elif from_date:
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        min_date = remove_date_dash(from_date)
        mega_path = glob.glob(pathname)
        
        files_to_ignore = []
        for file in mega_path:
            date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
            if date < min_date:
                files_to_ignore.append(file)

        for element in files_to_ignore:
            if element in mega_path:
                mega_path.remove(element)

    # If only to_date is provided, use all dates before that
    elif to_date:
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        max_date = remove_date_dash(to_date)
        mega_path = glob.glob(pathname)
        
        files_to_ignore = []
        for file in mega_path:
            date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
            if date > max_date:
                files_to_ignore.append(file)

        for element in files_to_ignore:
            if element in mega_path:
                mega_path.remove(element)
    
    # If neither argument is given, run on all available files
    else:
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        mega_path = glob.glob(pathname)
    
    if not test_limit:
        print(f"Ignoring {len(files_to_ignore)} files and going through {len(mega_path)} files based on from_date and to_date")
    
    return mega_path

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def extract_data(keyword_list:list, 
                 data_prefix: str, 
                 mega_path_pre:str,
                 root_path: str, 
                 from_date:str, 
                 to_date:str): 
    """Main function that runs and logs extraction of data
    """
    print("START data extraction for keywords: ", keyword_list, "\n")
    
    output_name = os.path.join(f'{root_path}', f'{data_prefix}_files', f'{data_prefix}_final.csv')
    print("Does the file already exist?: ", path.exists(output_name))
    
    if path.exists(output_name):
        mega_path = ignore_dates_less_than(output_name, mega_path_pre)
    else:
        mega_path = mega_path_pre
    print("Go through files: \n")     
    mega_path.sort()
    
    ic(mega_path)
    
    # Create a directory for these files
    # temp_path = root_path + "tmp_" + data_prefix + "/"
    temp_path = os.path.join(root_path, f'tmp_{data_prefix}')

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
            filename = os.path.join(temp_path, f'{data_prefix}_{file_name}.csv')
            df.to_csv(filename, index = False)

            print("Save of " + file_name + " done")
        else:
            print("Not enough data")
        print("-------------------------------------------\n")
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
    
    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    to_date = date.today().strftime("%Y-%m-%d")
    test_limit = None if test_limit == 'None' else test_limit
    small = literal_eval(small)

    return keywords, test_limit, from_date, to_date, language

########################################################################################################################
##     INPUT
########################################################################################################################

if __name__ == "__main__":
    print("---------- Running extract_data.py ----------")
    keywords, test_limit, from_date, to_date, language = main(sys.argv[1:])
    ori_keyword_list = keywords.split(",")
    print(f"Extracting data using '{ori_keyword_list}' from {from_date} to {to_date}, language = {language}")
    
    keyword_list = []
    for keyword in ori_keyword_list:
        if re.findall("~#", keyword):
            keyword = re.sub('~', '', keyword)
        else:
            keyword = re.sub("~", " ", keyword)
        keyword_list.append(keyword)
    
    data_prefix = keyword_list[0]

    if language == 'da':
        mega_path = define_megapath(test_limit, from_date, to_date)
    
    root_path = os.path.join("..") 

    ###############################
    print("------Creating folders------")
    files_path = os.path.join(root_path, f'{data_prefix}_files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
        print(f'Made folder: {files_path}')
    
    fig_path = os.path.join(root_path, "fig", data_prefix)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print(f'Made folder: {fig_path}')
    
    print('Done creating folders')
    
    print("--------Data extraction--------")
    if language == 'da':
        extract_data(keyword_list, data_prefix, mega_path, root_path, from_date, to_date)
    if language == 'en':
        extract_en_data(data_prefix, root_path)
