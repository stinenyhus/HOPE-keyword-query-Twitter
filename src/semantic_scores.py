"""
This won't run with my source but activate this instead:
source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate

Get Vader sentiment scores for texts
"""

import text_to_x as ttx
import pandas as pd
import getopt, sys
import re
import glob

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################

def semantic_scores(data_prefix: str, 
                    root_path:str):
    """
    data_prefix: indicates which dataset it is
    root_path: path to where the data is saved to
    """
    filename = root_path + data_prefix + "_data_pre.csv"
    filename = f'{root_path}{data_prefix}_files/{data_prefix}_data_pre.csv'
    sent_df = pd.read_csv(filename)
    print(sent_df.head())
    
    sent_df["mentioneless_text"] = sent_df["mentioneless_text"].astype(str)

    print("Conducting SA with VADER")
    print(sent_df.head())
    tts = ttx.TextToSentiment(lang='da', method="dictionary")
    out = tts.texts_to_sentiment(list(sent_df['mentioneless_text'].values))
    sent_df = pd.concat([sent_df, out], axis=1).dropna()
    print("Joining SA results")

    # filename_out = root_path + data_prefix + "_vis.csv"
    filename_out = f'{root_path}{data_prefix}_files/{data_prefix}_vis.csv'
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
        opts, args = getopt.getopt(argv,"hk:f:t:l:s:")
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
        elif opt in "-s":
            small = arg
            print('Small: ', small)
    print('Input keywords are ', keywords)
    return keywords

########################################################################################################################
##     INPUT
########################################################################################################################
    
if __name__ == "__main__":
    
    keywords = main(sys.argv[1:])
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
    root_path = "/home/commando/stine-sara/HOPE-keyword-query-Twitter/"
    
    ############################
    print("---------SENTIMENT ANALYSIS----------")
    semantic_scores(data_prefix, root_path)
