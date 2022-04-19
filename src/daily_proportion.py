'''
Take all the keywords from selvtest query and calculate daily what proportion of Danish tweets mention them that week
'''

import ndjson, os, sys
from csv import writer
from glob import glob
import pandas as pd
from typing import Optional, List
import re

from extract_data import remove_date_dash, main
from preprocess_stats import remove_quote_tweets
from smooth_and_entropy import gaussian_kernel_mapper

from visualize import set_base_plot_settings, set_late_plot_settings
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


def make_date_dash(date: str):
    '''
    takes in str in format '%Y%m%d' and returns '%Y-%m-%d'
    '''
    return date[:4] + '-' + date[4:6] + '-' + date[6:]

def ndjson_gen(path: str):
    for in_file in glob(path):
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def len_query(path: str, date: Optional[str]=None):
    '''
    Generator that yields the number of rows in the dfs in the path
    with the option of setting a date

    Args:
        path (str): path for the csv files to give to glob
        date (Optional[str]): Optional argument if you only want 
                            to include rows from a certian date
    
    returns int that is number of rows
    '''
    for in_file in glob(path):
        df_ = pd.read_csv(in_file, usecols=['date'])
        if date:
            df_ = df_[df_['date'] == make_date_dash(date)]
        yield len(df_.index)


def id_query(path: str, files_to_ignore: List[str]=[], date: Optional[str]=None):
    '''
    Generator that yields the number of rows in the dfs in the path
    with the option of setting a date

    Args:
        path (str): path for the csv files to give to glob
        files_to_ignore (List[str]): Optional argument of list of files in the path that should be ignored
        date (Optional[str]): Optional argument if you only want to include rows from a certian date
    
    returns list of all unique twitter ids
    '''
    for in_file in glob(path):
        if in_file in files_to_ignore:
            continue
        df_ = pd.read_csv(in_file, usecols=['date', 'id'])
        if date:
            df_ = df_[df_['date'] == make_date_dash(date)]
        yield list(df_['id'].unique())


def get_tweets_date(date: str):
    '''
    Return df with all tweets from a certain date

    Args:
        date (str): the date you want the tweets from. Should be in format '%Y-%m-%d'
    
    Return:
        pd.dataframe
    '''
    date = remove_date_dash(date)
    path = os.path.join('/data', '001_twitter_hope', 'preprocessed', 'da', f'td_{date}*.ndjson')

    created_at = []
    id = []
    text = []

    for tweet in ndjson_gen(path):
        if not tweet:
            continue
        created_at.append(tweet['created_at'])
        id.append(tweet['id'])
        text.append(tweet['text'])

    d = {'created_at': created_at, 'id': id, 'text': text}
    df = pd.DataFrame(data=d)
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime('%Y%m%d')
    df = df[df['date']==date]
    return df


def preprocess_df(df: pd.DataFrame):
    '''
    Function for preprocessing df
    '''
    df = df.sort_values(by='date').reset_index(drop=True)
    df = remove_quote_tweets(df)
    return df


def daily_proportion(date: str, data_prefix: str):
    query_path = os.path.join('..', f'{data_prefix}_files', f'{data_prefix}_all_ids.csv')
    files_to_ignore = []

    id_list = []
    for ids in id_query(query_path, files_to_ignore=files_to_ignore, date=date):
        id_list += ids
    covid_tweets = len(set(id_list))

    df = get_tweets_date(date)
    df = preprocess_df(df)

    all_tweets = len(df)
    proportion = covid_tweets/all_tweets
    return [make_date_dash(date), proportion, covid_tweets, all_tweets]


def proportion_and_write(path_date_dataprefix: tuple):
    path, date, data_prefix = path_date_dataprefix
    row = ['NaN'] + daily_proportion(date, data_prefix)
    append_list_as_row(path, row)

    if date[-2:] == '01':
        print('date = ', date)


def main_proportion(save_path: str, data_prefix: str, smooth: bool=False):
    import multiprocessing
    ### Creating out csv ###
    print('>> creating out data frame')
    out = pd.DataFrame(columns = ["date", "proportion", "n_keyword_tweets", "n_all_tweets"])
    out.to_csv(save_path)

    ## find the dates ##
    mega_path = glob(os.path.join('/data', '001_twitter_hope', 'preprocessed', 'da', f'*.ndjson'))
    dates = []
    for file in mega_path:
        d = re.findall(r'\/data\/001_twitter_hope\/preprocessed\/da\/td_(\d*)', file)[0]
        dates.append(d)
    print(f'latest date found is {max(dates)}')

    ## map to list of dates ##
    savepath_dates_dataprefix = [(save_path, date, data_prefix) for date in sorted(dates)]
    print('>> mapping function')
    pool = multiprocessing.Pool(processes=10)
    pool.map(proportion_and_write, savepath_dates_dataprefix)

    # smoothing
    if smooth:
        df = pd.read_csv(save_path)
        df = df.sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        print('>> smoothing')
        df['proportion_s5'] = gaussian_kernel_mapper(df['proportion'], sigma = 1, fwhm = 5)
        df['proportion_s20'] = gaussian_kernel_mapper(df['proportion'], sigma = 1, fwhm = 20)
        df.to_csv(f'{save_path[:-4]}_smoothed.csv')
        return df
    print('>> DONE creating csv <<')


## functions for visualizations ##

def plot_dailyproportion(df: pd.DataFrame, smoothing_value: int, save_folder: str, event_dict: Optional[dict]=None):
    '''
    visualize daily proportion
    '''
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)

    ax1 = sns.lineplot(x="date", y="proportion_percentage", 
                    color = palette[0], 
                    alpha=0.6,
                    linewidth = 2, data = df)
    ax1 = sns.lineplot(x="date", y=f"proportion_percentage_s{smoothing_value}", 
                    color = palette[5], 
                        linewidth = 5, data = df)
    
    if event_dict:
        x = df['date']
        y_smoothed = df[f"proportion_percentage_s{smoothing_value}"]
        for event, time in event_dict.items():
                plt.plot(time, y_smoothed[x == time],
                        marker='o', markersize=20, markeredgecolor="black", markerfacecolor="None")
                plt.text(time, # 0.007, 
                    y_smoothed[x == time] + 0.005,
                    s = event,
                    fontdict = dict(color="black",size=15),
                    bbox = dict(facecolor="white",alpha=0.5),
                    rotation=90)

    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    ax1.set_title("Proportion of tweets with COVID-19 keywords", fontdict = {'fontsize':50}, color = "Black")
    ax1.set_ylabel("%", fontdict = {'fontsize':30})

    if event_dict:
        plot_name = os.path.join(save_folder, f'dailyproportion_events_{smoothing_value}.png')
    else:
        plot_name = os.path.join(save_folder, f'dailyproportion_{smoothing_value}.png')
    fig.savefig(plot_name)


def visualize_proportion(df: pd.DataFrame, save_folder: str, event_dict: Optional[dict]=None):
    '''
    visualize daily proportion of tweets

    Args:
        df (pd.DataFrame): data frame with daily proportion
        save_folder (str): path for saving visualizations
        event_dict (Optional[dict]): optional argument if you want visualization with events
    
    '''
    print('>> Visualize daily proportion')

    df["date"] = pd.to_datetime(df["date"])
    df['proportion_percentage'] = df['proportion']*100
    smoothing_values = [5, 20]

    for smoothing_value in smoothing_values:
        # add percent to smoothed cols
        df[f'proportion_percentage_s{smoothing_value}'] = df[f"proportion_s{smoothing_value}"]*100
        # daily proportion plot
        plot_dailyproportion(df, smoothing_value, save_folder)
        # with events
        if event_dict:
            plot_dailyproportion(df, smoothing_value, save_folder, event_dict)
    print(f'plots saved in folder: {save_folder}')


if __name__=='__main__':
    save_folder = os.path.join('..', 'daily_proportion_files')
    file_name = 'daily_proportion.csv'
    save_fig_folder = os.path.join('..', 'data_da')
    save_path = os.path.join(save_folder, file_name)
    smooth = True

    print(f'''---------- Running daily_proportion.py ---------- 
    save_path = {save_path}
    smooth = {smooth},
    save_fig_folder = {save_fig_folder}''')

    keywords, test_limit, from_date, to_date, language, day_prob = main(sys.argv[1:])
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

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_fig_folder):
        os.mkdir(save_fig_folder)
    
    main_proportion(save_path, data_prefix, smooth=smooth)

    # visualize only if smoothed
    if smooth:  # only happens if smooth is True
        col_list = ['date', 'proportion', 'n_keyword_tweets', 'n_all_tweets', 'proportion_s5', 'proportion_s20']
        df = pd.read_csv(f'{save_path[:-4]}_smoothed.csv', usecols=col_list)
    
        event_dict = {
                "first lockdown": dt.datetime(2020, 3, 11),
                "Queens speech": dt.datetime(2020, 3, 17),
                #   "phase 1 reopning": dt.date(2020, 4, 14),
                "mask in public transport": dt.datetime(2020, 8, 22),
                "meet-up limit 50-10": dt.datetime(2020, 10, 23),
                "mink": dt.datetime(2020,11,4),
                "second lockdown": dt.datetime(2020, 12, 16),
                "first vaccine": dt.datetime(2020,12,27),
                "Pfizer approved": dt.datetime(2021, 1, 6),
                "lockdown extended": dt.datetime(2021, 1, 28),
                "AZ paused": dt.datetime(2021, 3, 11),
                "AZ withdrawn": dt.datetime(2021, 4, 4),
                "coronapas launced": dt.datetime(2021, 5, 27),
                "announced restrictions-lifting": dt.datetime(2021, 8, 27),
                "all restrictions lifted": dt.datetime(2021, 9, 10),
                "restriction lifts annonced": dt.datetime(2022, 1, 17),
                "all restrictions lifted": dt.datetime(2022, 2, 1)
                }
        
        visualize_proportion(df, save_fig_folder, event_dict)

