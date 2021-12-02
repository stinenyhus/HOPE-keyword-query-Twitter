'''
reads in gold standard ndjson
writes csv with tweets including keywords
'''
import ndjson
from csv import writer
import pandas as pd
import re
import os
import time
from typing import List, Optional


## define functions ##
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def ndjson_gen(filepath: str):
    with open(filepath) as f:
        reader = ndjson.reader(f)

        for post in reader:
            yield post


def identify_bots(df: pd.DataFrame, text_col: str):
    """
    Identifies bot-like tweets/quote tweets, when 50 first characters are the exact same, remove as duplicates
    NB: slightly modifies version of Maris' function in 'preprocess_stats.py'
    
    Args:
        df (pd.DataFrame): Dataframe
        text_col (str): column in dataframe to look for duplicates in

    return:
        df: pandas DataFrame with column "dupe50"
    """
    df["text50"] = df[text_col].str[0:50]
    
    df["dupe50"] = df["text50"].duplicated(keep = "first")
    
    return df.drop(["text50"], axis=1)


def ndjson_to_csv(in_file: str,
                  out_file: str, 
                  cols: List[str], 
                  cols_in_csv: Optional[List[str]]=None,
                  keywords: Optional[List[str]]=None,
                  cols_within_cols: Optional[List[str]]=None,
                  rm_mentions: Optional[bool]=False):
    '''
    converts ndjson files to csv
    
    args:
        in_file (str): name of the input ndjson file
        out_file (str): name of the output csv file
        cols (List[str]): columns wanted in the output file (must be contained in all of the ndjson)
        cols_in_csv (Optional[List[str]]): manually specifying the names of the columns in the csv
        keywords (Optional[List[str]]): if files with certain keywords should be included in the csv
        cols_within_cols (Optional[List[str]]): list of keys containing a new dictionary, in which all values should be included in csv
        rm_mentions (Optional[bool]): true if mentions to be removed from text 
    
    '''
    in_filepath = os.path.join('..','..','data', f'{in_file}.ndjson')
    out_filepath = os.path.join('..','..','data', f'{out_file}.csv')
    
    #creating out csv
    if cols_in_csv:
        out = pd.DataFrame(columns = cols_in_csv)
    else:
        out = pd.DataFrame(columns = cols)
    out.to_csv(out_filepath, encoding="utf-8")
    
    start_time = time.time()
    j = 0
    for i, post in enumerate(ndjson_gen(in_filepath)):
        # check for keywords
        cont = True
        for keyword in keywords:
            if keyword in post['text']:
                cont = False
        if cont:
            continue
        
        row = [j] + [post[col] for col in cols]

        if cols_within_cols:
            row += [value for col in cols_within_cols for value in post[col].values()]
        if rm_mentions:
            mention = ' '.join(re.findall(r'@\w+', post['text']))
            text_mentionless = re.sub(r'@\w+\s*', r'', post['text'])
            row += [mention, text_mentionless]

        assert len(row) == len(out.columns) + 1  # plus 1 due to the index 
        append_list_as_row(out_filepath, row)
        j +=1
        
        mid_time = time.time()
        if i % 100 == 0: 
            print(f'Running model on row number {i} now finished - time in min: {(mid_time - start_time)/60}')


def remove_bot_tweets(in_file: str, out_file: str):
    '''
    reads in csv and writes two csv files: One without all bot tweets and one including only
    bot tweets. 

    Args:
        in_file (str): name of the file being read in
        out_file (str): name of the output file (where the csv incl. bot tweets will be named '..._bots.csv')
    '''
    print('removing bot tweets')
    in_filepath = os.path.join('..','..', 'data', f'{in_file}.csv')
    print('reading csv')
    df = pd.read_csv(in_filepath, index_col=[0])

    print('identifying bots')
    df = identify_bots(df, 'text_mentionless')

    df_nobots = df[df["dupe50"] == False].reset_index().drop(["dupe50", "index"], axis=1)
    df_bots = df[df["dupe50"] == True].reset_index().drop(["dupe50", "index"], axis=1)

    print(f'writing csv: {out_file}.csv and {out_file}_bots.csv')
    df_nobots.to_csv(os.path.join('..','..','data', f'{out_file}.csv'))
    df_bots.to_csv(os.path.join('..','..','data', f'{out_file}_bots.csv'))


if __name__=='__main__':
    keywords = ["vaccin"]
    in_file = 'vaccin_gold_standard_2019-01-01_2021-10-08'
    cols = ['id', 'lang', 'text', 'created_at', 'author_id']
    cols_in_csv = cols + ['retweet_count', 'reply_count', 'like_count', 'quote_count',
                          'mentions', 'text_mentionless']
    cols_within_cols = ['public_metrics']
    out_file = 'goldstandard_engagementdata_all'
    
    # ndjson_to_csv(in_file, out_file, cols, 
    #               cols_in_csv=cols_in_csv, 
    #               keywords=keywords, 
    #               cols_within_cols=cols_within_cols,
    #               rm_mentions=True)
    
    remove_bot_tweets(out_file, out_file[:-4])
