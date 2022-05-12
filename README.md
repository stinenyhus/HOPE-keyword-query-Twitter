# Pipeline for querying Twitter by keyword(s)

### Authors
- Maris Sala
- Sara Møller Østergaard
- Stine Nyhus Larsen

## Examples
An example Jupyter Notebook visualizing the different parts of the code has been added. Check out how mentions of different keywords are represented on Twitter in ``notebooks/visualize_HOPE.ipynb``

## Usage

The pipeline can be run on two different datasets:

1. Querying Danish Twitter for keywords. Based on keywords (and possibly date specifics) the pipeline extracts tweets from our Danish Twitter corpus where the keywords match with texts.
2. Running pipeline on subset of tweets from English Twitter. The tweets have been obtained using already defined keywords: ``denmark AND (covid | corona | omicron)``. 

### Run the pipeline
Clone the repository and run the pipeline in one of two ways:

1. For query of a single keyword search, use the following approach and run the pipeline directly
```
cd src
bash pipeline.sh -k key &> logs/key_[today's_date].log
```

2. For query of multiple keyword searches, use the file ``run_queries.sh``; change the date in line 1 to today's date in the form DDMM (e.g. ``DATE=1502``), uncomment the relevant lines and run it in the terminal, possibly in a screen, as
```
cd src
bash run_queries.sh
```
Log-files are automatically saved in the same folder as when using approach 1. 

To run the pipeline an entry for the key must exist in the config file ``src/keyword_config.ini``. To add a new entry the following must be provided

| Name  | Meaning  | Format  | Example1  | Example2  |
|---|---|---|---|---|
| key  | Name of the entry. Using the same name as the first of the keywords is reccommended.  | name | covid  | denmark-omicron  |
| keywords  | Keyword(s) to query (in Danish tweets) or used to query (in English tweets). The first keyword will be used as dataprefix for folders and files created from running the pipeline.  | keyword1,keyword2  | covid  | covid,dkpol  |
| from_date  | From date: if one wants to specify date range. If from_date is None, the code queries for data up until the earliest dates. | YEAR-MONTH-DAY or None  | 2020-01-01  | None  |
| to_date  | To date: if one wants to specify date range. If to_date is None, the code queries for data up until the latest dates.  | YEAR-MONTH-DAY or None  | 2020-01-30  | None  |
| test_limit  | Test with limit: to speed up testing, samples only from data of this year/month/day  | YEARMONTHDAY or None  | 202001  | None |
| small | Small or not: most datasets are small (100-500 tweets per day), use False when it's a large dataset (1000-more tweets per day). This is used for setting the parameters for Gaussian smoothing. Produces two types of smoothing plots (smoother and less smoother, so that smoothing can be done automatically).  | Boolean | True | False |
| lan  | Language: whether to run the pipeline for Danish or English Twitter.  | Language  | da  | en |
| daily_proportion | whether the query is run to calculate the daily proportion | Boolean | True | False

*If keywords are special:*
- If keywords include hashtags trail the 1st hashtag with ~. E.g. ```keywords = keyword1,~#keyword2```.
- If keywords include words with spaces replace space with ~. E.g. ```keywords = keyword1,key~word2```.



## Description of steps in the pipeline
1. Extract data
```
src/extract_data.py
```
If you are running the pipeline with ``lan = da``, this script extracts data from ```/data/001_twitter_hope/preprocessed/da/*.ndjson``` - this includes all preprocessed Danish Twitter data. Creates a file with matches with keywords per file and saves them to ``tmp_keyword/`` folder which it creates itself (allows for running the code for different keywords simultaneously because each keyword query uses a separate data folder). If you are running the pipeline with ``lan = en``, this script extracts data from ```/data/twitter-omicron-denmark/*``` and saves them as ``keyword1_data.csv``.

2. Join files
```
src/join_files.py
```
This script only runs if ``lan = da``. Joins together all files starting with keyword1 in the data folder, joins them together as ``keyword1_data.csv``. Deletes all the keyword files and the created temporary data folder to reduce taking up space.

3. Preprocess stats
```
src/preprocess_stats.py
```
Preprocesses the data: cleans tweets from mentions, hashtags, emojis, URLs (adds a cleaned tweet column, keeps the original tweet intact). Removes quote tweets from the data set. Outputs statistics. These are captured in the ``src/logs/``. Outputs ``keyword1_data_pre.csv``.

4. Sentiment BERT
```
src/sentiment_bert.py
```
Classifies the tweets with a sentiment label (i.e. positive, negative, neutral) using a BERT model. Corresponding with the label a polarity score is assigned to each tweets (positive = 1, negative = -1, neutral = 0). Outputs ``keyword1_data_bert.csv``.

5. Semantic scores
```
src/semantic_scores.py
```
Calculates semantic scores with Vader for the cleaned tweets. Outputs ``keyword1_vis.csv``.

6. Smoothing
```
src/smooth_and_entropy.py
```
Gaussian smoothing on number of tweets per day and compound scores (can calculate entropy as well). Good for clarifying the visuals. Smoothing values are defined by the value for ``small`` in the config file. Outputs ``keyword1_smoothed.csv``.

7. Visualize
```
src/visualize.py
```
Creates initial visuals: keyword mentions frequency over time, compound sentiment over time, polarity sentiment over time, frequent hashtags, frequent words, wordcloud, bigram graphs with k varying between 1 and 5. Saves to ``fig/dataprefix/``

*NOTE: if the file for the specific keyword search has already been conducted, the code first checks whether that is true, and only adds the data for new incoming dates, instead of rerunning extraction and preprocessing on all of the data.*


## Covid timeline
The repository also includes a timeline over important events in the Covid pandemic (`timeline_covid.xlsx`) as identified by SSI (Statens Serum Institut):

 https://www.ssi.dk/-/media/arkiv/subsites/covid19/presse/tidslinje-over-covid-19/covid-19-tidslinje-for-2020-2022-lang-version---version-1---april-2022.pdf?la=da.


All events in the timeline have been annotated with three different codes:

* type: what is the type of the event described?
    * epidemiological: number of confirmed cases, new variant detected, number of people vaccinated etc. 
    * policy: changes regarding restrictions, vaccine authorizations, masks, etc.
    * other: events not fitting either of the two. Examples are the queen's speech, changes in test capacity
* nationality: nationality of the event
    * danish: event happens in Denmark.
    * international: event is global/international. This can for example be global EU restrictions (closing all borders) or when the first case in another country is confirmed
* relevant: is the event relevant/impactful in the context of news/social media data
    * 0: the event is not relevant
    * 1: the event is relevant
    * 2: the event may be relevant

All events have been coded independently by Sara Møller Østergaard and Stine Nyhus Larsen. Cases of disagreement have been discussed and cleared with Rebekah Baglini until agreement was reached. 
    
