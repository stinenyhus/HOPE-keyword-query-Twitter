"""
Join files made by 00_extract_data.py from data/ folder with the same prefix
"""
import re
import glob
import pandas as pd
import getopt, sys
import os, shutil

### Define Functions ###

def get_df(filenames):
    df = pd.read_csv(filenames[0], header=None)

    for file in filenames[1:]:
        df_0 = pd.read_csv(file, header = None)
        df = df.append(df_0)

    df = df.drop_duplicates()

    return df

def clean_dates(df):
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def join_files(data_prefix):
    filename = "../data/" + data_prefix + "*.csv"
    filenames = glob.glob(filename)
    output_name = "../" + data_prefix + "_data.csv"
    
    print("Get data: ", data_prefix)
    df = get_df(filenames)

    print("Save file")
    df.to_csv(output_name, index=False)
    del df

def empty_data_folder():
    print("Delete contents of data/")
    folder = '/home/commando/maris/hope-keyword-templates/data/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

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
    return keywords#, test_limit, from_date, to_date - these are not necessary to output for join_files.py

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
    
    ###############################
    print("--------JOIN FILES--------")
    join_files(data_prefix)
    empty_data_folder()
    print("--------FINISHED----------")