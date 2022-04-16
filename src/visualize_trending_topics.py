import pandas as pd
import ndjson
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import plotly.express as px

def get_week_list_from_range(start, end):
    """Get a list of weeks between two input weeks

    Args:
        start (str): On the form 'ww-YYYY'
        end (str): On the form 'ww-YYYY' - must be larger than start

    Returns:
        weeks: list of weeks between the two input dates
    """    
    s = start.split("-")
    e = end.split("-")

    # last week to add - used for previous topic of week 01 of a year
    if s[0] == "01":
        last_elm = ["52"+f'-{int(s[1])-1}']
    else:
        last_elm = [str(int(s[0])-1)+f'-{s[1]}']
    
    # If years are chosen from the same year - only possibility at the moment
    if s[1] == e[1]:
        weeks = [f'{w}-{s[1]}' for w in range(int(s[0]), int(e[0])+2)] + last_elm
        return [f'0{string}' if len(string)==6 else string for string in weeks]
    else:
        if s[1] == "2020" and e[1] == "2021":
            first_half = [f'{w}-{s[1]}' for w in range(int(s[0]), 54)]
            second_half = [f'{w}-{e[1]}' for w in range(1, int(e[0])+1)]
            weeks = first_half + second_half + last_elm
        elif s[1] == "2021" and e[1] == "2022":
            first_half = [f'{w}-{s[1]}' for w in range(int(s[0]), 53)]
            second_half = [f'{w}-{e[1]}' for w in range(1, int(e[0])+1)]
            weeks = first_half + second_half + last_elm
        return [f'0{string}' if len(string)==6 else string for string in weeks]

def get_prev_week(week):
    """Get the previous week of any week

    Args:
        week (str): Week to get previous week from, on the form 'ww-YYYY'

    Returns:
        week (str): The previous week also on the form 'ww-YYYY'
    """    
    week_split = week.split("-")
    # Taking care of weeks around new years 
    if week == "01-2021":
        return "53-2020"
    elif week == "01-2022":
        return "52-2021"
    
    # Taking care of weeks below 10 as they are rendered weirdly
    string = str(int(week_split[0])-1)+f'-{week_split[1]}'
    if len(string) == 6:
        return f'0{string}'
    
    # If none of those cases, return the string 
    else: 
        return string


def get_position(terms):
    d = pd.DataFrame()
    for week in terms:
        df = pd.DataFrame({"Word": terms[week].keys(), "Frequency": terms[week].values()})
        df["Week"] = week
        df = df.sort_values('Frequency', ascending = False)
        df["Position"] = list(range(1, len(df.index)+1))
        d = pd.concat([d, df])
    return d

def position_change(df: pd.DataFrame, column: str = "Position"):
    word_dict = {week:{word:pos for word,pos in zip(df[df["Week"]==week]["Word"], df[df["Week"]==week][column])} \
    for week in df["Week"].unique()}
    df[f"Last_{column.lower()}"] = pd.NA
    for week in df["Week"].unique()[:-1]:
        now = df[df["Week"]==week]
        last_week = get_prev_week(week)
        no_pos = len(now.index)+1
        df[f"Last_{column.lower()}"][df["Week"]==week] = [word_dict[last_week].get(word, no_pos) for word in now["Word"]]
    df[f"{column}_change"] = df[f"Last_{column.lower()}"]-df[column]
    df[f"Relative_{column.lower()}_change"] = df[f"{column}_change"]/df[column] * 100
    return df


def get_trending_detrending_terms(df, n_terms, by = "Position_change"):
    trending = pd.DataFrame()
    detrending = pd.DataFrame()
    for week in df["Week"].unique():
        sub = df[df["Week"] == week]
        # sub["abs"] = abs(sub["Position_change"])
        trend = sub.sort_values(by, ascending = False)
        trend = trend[:n_terms]
        detrend = sub.sort_values(by, ascending = True)
        detrend = detrend[:n_terms]
        trending = pd.concat([trending, trend])
        detrending = pd.concat([detrending, detrend])
    return (trending, detrending)

def plot_trending(df, title, by = "Position_change"):
    fig = px.line(df, 
                  x="Week", 
                  y=by, 
                  color="Word", 
                  markers=True,
                  title=title)
    fig.update_yaxes(title_text="Change in position relative to last week")
    fig.update_traces(marker_size=8)
    return fig

def visualize_trending_topics(start: str, end: str, n_topics: int = 3, by = "Position_change"):
    '''
    visualizes trending and de-trending topics from the term_freq_all.ndjson file

    Args:
        start (str): first week
        end (str): last week

    returns tuple with two plotly plots
        trend_plot, detrend_plot
    '''
    with open('/home/hope-twitter-analytics/hope_twitter_analytics/data_da/term_freq_all.ndjson', 'r') as f:
        data = ndjson.load(f)
    
    weeks = get_week_list_from_range(start, end)
    # weeks = [f'0{week}' if int(week[0]) in range(10) else week for week in weeks]
    # weeks = [week for week in weeks if week[0]!='0']
    terms = {key: data[0][key] for key in weeks}

    d = get_position(terms)
    w_change = position_change(d)

    trending, detrending = get_trending_detrending_terms(w_change, n_topics)
    trending = trending[trending["Week"]!= weeks[-1]]
    detrending = detrending[detrending["Week"]!= weeks[-1]]

    trend_plot = plot_trending(trending, "Trending topics", by = by)
    detrend_plot = plot_trending(detrending, "Detrending topics", by = by)

    return trend_plot, detrend_plot


if __name__ == "__main__":    
    with open('term_freq_all.ndjson', 'r') as f:
        data = ndjson.load(f)

    start = '51-2020'
    end = '04-2021'
    
    weeks = get_week_list_from_range(start, end)
    terms = {key: data[0][key] for key in weeks}

    d = get_position(terms)
    w_change = position_change(d)

    trending, detrending = get_trending_detrending_terms(w_change, 3, by = "Position_change")
    trending = trending[trending["Week"]!= weeks[-1]]
    detrending = detrending[detrending["Week"]!= weeks[-1]]

    trend_plot = plot_trending(trending, "Trending topics", by = "Position_change")
    trend_plot.show()
    detrend_plot = plot_trending(detrending, "Detrending topics", by = "Relative_change")
    detrend_plot.show()
