"""
Extract hashtags per day for a dataset.
Output:
Hashtag | date | nr_of_mentions of this hashtag
"""

import pandas as pd
from icecream import ic
import re

def extract_hashtags(row):
    unique_hashtag_list = list(re.findall(r'#\S*\w', row["text"]))
    return unique_hashtag_list

def hashtag_per_row(data):
    # Create hashtags column with the actual unique hashtags
    data["hashtags"] = data.apply(lambda row: extract_hashtags(row), axis = 1)

    # Let's take a subset of necessary columns, add id
    df = data[["created_at", "hashtags", "id"]]
    df["id"] = df["id"].astype(str)
    
    # Hashtag per row
    # convert list of pd.Series then stack it
    df = (df
     .set_index(['created_at','id'])['hashtags']
     .apply(pd.Series)
     .stack()
     .reset_index()
     .drop('level_2', axis=1)
     .rename(columns={0:'hashtag'}))
    #lowercase!
    df["hashtag"] = df["hashtag"].str.lower()
    df["hashtag"] = df["hashtag"].str.replace("'.", "")
    df["hashtag"] = df["hashtag"].str.replace("’.", "")

    return df

df = pd.read_csv("../all_vis.csv", dtype={"id":str})
df = df.sort_values(by='created_at').reset_index(drop=True)
ic(len(df))

freq_hashtags = hashtag_per_row(df)

freq_hashtags.to_csv("all_hashtags_per_tweet.csv", index = False)