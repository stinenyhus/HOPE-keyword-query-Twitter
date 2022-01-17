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
from configparser import ConfigParser

########################################################################################################################
##     DEFINE FUNCTIONS
########################################################################################################################

def remove_retweets(data):
    """Finds tweets that are RTs and removes them
    data: pandas DataFrame with (at leat) column "text"
    """
    patternDel = "^RT"
    data["text"] = data["text"].astype(str)
    filtering = data['text'].str.contains(patternDel)
    removed_RT = data[~filtering].reset_index(drop=True)
    return removed_RT

def extract_keywords(row, 
                     keyword_list:list):
    """Lowercases tweets, finds keyword matches
    row: pandas DataFrame row
    keyword_list: list of keywords (str)
    """
    #row["text"] = row["text"].astype(str)
    tweet = row["text"].lower()
    res = [ele for ele in keyword_list if(ele in tweet)] 
    return res

def remove_date_dash(text: str):
    """Removes dash in dates when ignoring already processed files based on whether the date exists or is new
    text: date as a string
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


def ignore_dates_less_than(output_name):
    """Finds out which dates already exist in the processed dataset and removes files from mega path that are less than the max date
    output_name: path to the already existing output dataframe
    """
    ori_df = pd.read_csv(output_name)
    ori_df = ori_df[ori_df["0"] != 'created_at'].reset_index(drop=True)
        
    dates = pd.to_datetime(ori_df["0"].dropna()[1:], utc=True).dt.strftime('%Y-%m-%d').drop_duplicates().reset_index(drop = True).astype(str)
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

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def extract_data(keyword_list:list, 
                 data_prefix: str, 
                 mega_path:str,
                 root_path: str, 
                 from_date:str, 
                 to_date:str):
    """Main function that runs and logs extraction of data
    """
    print("START data extraction for keywords: ", keyword_list)
    print("---")
    
    # output_name = root_path + data_prefix + "_data.csv"
    output_name = f'{root_path}{data_prefix}_files/{data_prefix}_data.csv'
    print("Does the file already exist?: ", path.exists(output_name))
    
    if path.exists(output_name):
        mega_path = ignore_dates_less_than(output_name)
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
            print(f'Running pipeline with key: {key}, keywords: {keywords} from {from_date} and small = {small}')
        # elif opt in "-f":
        #     from_date = arg
        #     print('Date specifics: from ', from_date)
        # elif opt in "-t":
        #     to_date = arg
        #     print(' to ', to_date)
        # elif opt in "-l":
        #     test_limit = arg
        #     print('TESTING: ', test_limit)
        # elif opt in "-s":
        #     small = arg
        #     print('Small: ', small)
    print('Input keywords are ', keywords)
    return keywords, test_limit, from_date, to_date

########################################################################################################################
##     INPUT
########################################################################################################################

if __name__ == "__main__":
    
    keywords, test_limit, from_date, to_date = main(sys.argv[1:])
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
    elif (len(from_date) > 1) & (len(to_date) > 1):
        ic(from_date, to_date)
        ic(type(from_date), type(to_date))
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        
        min_date = remove_date_dash(from_date)
        max_date = remove_date_dash(to_date)
        mega_path = glob.glob(pathname)
        
        print(min_date, max_date)
        
        files_to_ignore = []
        for file in mega_path:
            date = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
            if (date < min_date) or (date > max_date):
                files_to_ignore.append(file)

        ic(files_to_ignore)
        
        for element in files_to_ignore:
            if element in mega_path:
                mega_path.remove(element)
                
    elif len(from_date) > 1:
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
                
    elif len(to_date) > 1:
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
    else:
        # Runs through all the files here, date does not matter
        pathname = '/data/001_twitter_hope/preprocessed/da/*.ndjson'
        mega_path = glob.glob(pathname)
    
    root_path = os.path.join("..") 

    ###############################
    print("------CREATING FOLDERS------")
    files_path = os.path.join(root_path, f'{data_prefix}_files')
    if not os.path.exists(files_path):
        os.makedirs(files_path)
        print(f'Made folder: {files_path}')
    
    fig_path = os.path.join(root_path, "fig", data_prefix)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print(f'Made folder: {fig_path}')
    
    print('done creating folders')
    
    print("--------EXTRACT DATA--------")
    extract_data(keyword_list, data_prefix, mega_path, root_path, from_date, to_date) 