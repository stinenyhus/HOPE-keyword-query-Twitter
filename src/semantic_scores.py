"""
This won't run with my source but activate this instead:
source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate

Get Vader sentiment scores for texts
"""

import text_to_x as ttx
import pandas as pd
import getopt, sys, os
import re
import glob
from configparser import ConfigParser
from ast import literal_eval

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def semantic_scores(data_prefix: str, 
                    root_path:str,
                    language: str):
    """
    data_prefix: indicates which dataset it is
    root_path: path to where the data is saved to
    language: the language of the text
    """
    # filename = root_path + data_prefix + "_data_pre.csv"
    filename = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_data_bert.csv')
    sent_df = pd.read_csv(filename, lineterminator='\n')
    
    sent_df["mentioneless_text"] = sent_df["mentioneless_text"].astype(str)
    sent_df = sent_df.drop_duplicates()

    tts = ttx.TextToSentiment(lang=language, method="dictionary")
    out = tts.texts_to_sentiment(list(sent_df['mentioneless_text'].values))
    sent_df = pd.concat([sent_df, out], axis=1).dropna()

    # filename_out = root_path + data_prefix + "_vis.csv"
    filename_out = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_vis.csv')
    
    print(f"Does the file {filename_out} already exist?: {os.path.exists(filename_out)}")
    if os.path.exists(filename_out):
        ori_df = pd.read_csv(filename_out, lineterminator="\n")
        sent_df = pd.concat([ori_df, sent_df]) 

    print(sent_df.head())
    sent_df.to_csv(filename_out, index = False)

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

    return keywords, language

########################################################################################################################
##     INPUT
########################################################################################################################
    
if __name__ == "__main__":
    print("\n---------- Running semantic_scores.py ----------")
    keywords, language = main(sys.argv[1:])
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
    root_path = os.path.join("..") 
    
    ############################
    print(f"Running sentiment analysis on keywords {keyword_list} with language = {language}")
    semantic_scores(data_prefix, root_path, language)
