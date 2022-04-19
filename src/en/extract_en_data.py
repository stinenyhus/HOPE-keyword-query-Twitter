'''
Extracts relevant data from the english tweets
'''
import ndjson
from glob import glob
from csv import writer
import pandas as pd
import re
import os
from typing import List, Optional
import sys, getopt
from configparser import ConfigParser


## define functions ##
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def ndjson_gen(filepath: str):
    for file in glob(filepath):
        print(f'opening {file}')
        with open(file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post


def ndjson_to_csv(in_filepath: str,
                  out_filepath: str, 
                  cols: List[str], 
                  cols_in_csv: Optional[List[str]]=None,
                  keywords: Optional[List[str]]=None,
                  cols_within_cols: Optional[List[str]]=None,
                  rm_mentions: Optional[bool]=False,
                  rm_retweets: Optional[bool]=False):
    '''
    converts ndjson files to csv
    
    args:
        in_filepath (str): path of the input ndjson file
        out_filepath (str): path of the output csv file
        cols (List[str]): columns wanted in the output file (must be contained in all of the ndjson)
        cols_in_csv (Optional[List[str]]): manually specifying the names of the columns in the csv
        keywords (Optional[List[str]]): if files with certain keywords should be included in the csv
        cols_within_cols (Optional[List[str]]): list of keys containing a new dictionary, in which all values should be included in csv
        rm_mentions (Optional[bool]): true if mentions to be removed from text 
        rm_retweets (Optional[bool]): true if retweets are not to be included
    
    '''

    #creating out csv
    if cols_in_csv:
        out = pd.DataFrame(columns = cols_in_csv)
    else:
        out = pd.DataFrame(columns = cols)
    out.to_csv(out_filepath, encoding="utf-8")
    

    j = 0
    for i, post in enumerate(ndjson_gen(in_filepath)):
        if keywords:
            # check for keywords
            cont = True
            for keyword in keywords:
                if keyword in post['text']:
                    cont = False
            if cont:
                continue
        if rm_retweets:
            if "RT " in post['text'][:10]: # remove retweets
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



def extract_en_data(data_prefix: str, root_path: str):
    path = os.path.join('/data', 'twitter-omicron-denmark', '*')
    out_filepath = os.path.join(root_path, f'{data_prefix}_files', f'{data_prefix}_data.csv')

    print(f'''Running extract_en_data with:
            data_prefix={data_prefix},
            path={path}''')

    ndjson_to_csv(in_filepath=path,
                  out_filepath=out_filepath,
                  cols = ['created_at', 'id', 'text'],
                  rm_retweets=True)
    
    print('--------------- \nextract_en_data.py done \n---------------')


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
            print(f'Running extract english data with key: {key}, keywords: {keywords} from {from_date}. Small = {small}. Language = {language}.')
    
    # convert make sure None is not a str
    from_date = None if from_date == 'None' else from_date
    to_date = None if to_date == 'None' else to_date
    test_limit = None if test_limit == 'None' else test_limit

    return keywords


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
    

    data_prefix = keyword_list[0]
    root_path = '..'

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

    extract_en_data(data_prefix, root_path)
