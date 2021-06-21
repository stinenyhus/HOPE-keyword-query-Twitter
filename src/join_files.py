"""
Join files made by 00_extract_data.py from data/ folder with the same prefix
"""
import re
import glob
import pandas as pd
import getopt, sys
import os, shutil
import os.path
from os import path

########################################################################################################################
##     DEFINE FUNCTIONS
########################################################################################################################

def get_df(filenames):
    df = pd.read_csv(filenames[0], header = None, sep=",")
    
    ####################################
    #df.columns = df.iloc[0]
    #df = df.iloc[1:]
    ####################################

    for file in filenames[1:]:
        print(file)
        df_0 = pd.read_csv(file, header = None, sep=",", lineterminator='\n')
        ####################################
        #df_0.columns = df_0.iloc[0]
        #df_0 = df_0.iloc[1:]
        ####################################
        df = df.append(df_0)

    df = df.drop_duplicates()
    
    print("DF:", df.head())

    return df

def clean_dates(df):
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def empty_data_folder(temp_path):
    print("Delete contents of tmp_*/")

    try:
        shutil.rmtree(temp_path)
    except OSError:
        print ("Deletion of the directory %s failed" % temp_path)
    else:
        print ("Successfully deleted the directory %s" % temp_path)

########################################################################################################################
##     MAIN FUNCTION
########################################################################################################################
        
def join_files(temp_path,data_prefix):
    filename = temp_path + "*.csv"
    filenames = glob.glob(filename)
    output_name = "../" + data_prefix + "_data.csv"
    
    print("Get data: ", data_prefix)
    df = get_df(filenames)
    df = df.rename(columns = {0:"0", 1:"1", 2:"2", 3:"3"})

    print("Does the file already exist?: ", path.exists(output_name))
    if path.exists(output_name):
        ori_df = pd.read_csv(output_name)
        df = pd.concat([ori_df, df])
    
    print("Save file")
    df.to_csv(output_name, index=False)
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
    temp_path = "/home/commando/maris/hope-keyword-templates/tmp_" + data_prefix + "/"
    
    ###############################
    print("--------JOIN FILES--------")
    if len(os.listdir(temp_path)) == 0:
        print("Directory is empty")
    else:    
        print("Directory is not empty")
        join_files(temp_path,data_prefix)
    empty_data_folder(temp_path)
    print("--------FINISHED----------")