'''
Script for creating csv with only date and tweet-id from preprocess_stats.py
This is used for the daily proportion
'''

import os, sys, re
import pandas as pd

from extract_data import main

def write_ids_csv(data_prefix: str):
    '''
    appends lines to '{data_prefix}_all_ids.csv' with date and id from 
    '{data_prefix}_data_pre.csv' from preprocess_stats.py

    Args:
        data_prefix (str): data_prefix to find the _data_pre csv
    '''
    filename = os.path.join("..", f"{data_prefix}_files", f"{data_prefix}_data_pre.csv")
    data_pre = pd.read_csv(filename, lineterminator='\n', usecols=['date', 'created_at', 'id'])
    out_path = os.path.join('..', f'{data_prefix}_files', f'{data_prefix}_all_ids.csv')

    if os.path.exists(out_path):  # if the file already exists
        old = pd.read_csv(out_path)
        data_pre = pd.concat([old, data_pre])
    
    data_pre.to_csv(out_path)



if __name__=='__main__':
    print("---------- Running all_covid_ids.py ----------")
    keywords, test_limit, from_date, to_date, language, daily_proportion = main(sys.argv[1:])
    ori_keyword_list = keywords.split(",")
    
    keyword_list = []
    for keyword in ori_keyword_list:
        if re.findall("~#", keyword):
            keyword = re.sub('~', '', keyword)
        else:
            keyword = re.sub("~", " ", keyword)
        keyword_list.append(keyword)
    
    data_prefix = keyword_list[0]

    new_data = os.path.join("..", f'{data_prefix}_files', f'{data_prefix}_data.csv')
    if not os.path.exists(new_data):
        quit()

    write_ids_csv(data_prefix)