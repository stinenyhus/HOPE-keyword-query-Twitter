# Pipeline for querying Twitter by keyword(s)

### Author
Maris Sala

## Usage

Based on keywords (and possibly date specifics) this pipeline extracts tweets from our Twitter corpus where the keywords match with texts.

```bash
cd src
nohup bash pipeline.sh -k keyword1,keyword2 -f 2020-12-01 -t 2020-12-30 &> logs/keyword1_logs.log &

```
Use "bash" and *not* "sh"!
Nohup allows for the code to run in the background while freeing up the terminal. It also saves the logs into the logs/ folder where one can later see statistics about the dataset as well as what might have gone wrong and where.

Usage without nohup:
```bash
cd src
bash pipeline.sh -k keyword1,keyword2 -f 2020-12-01 -t 2020-12-30

```

| Flag  | Meaning  | Format  | Example1  | Example2  |
|---|---|---|---|---|
| -k  | Keyword(s) to query  | keyword1,keyword2  | covid  | covid,#dkpol  |
| -f  | From date: if one wants to specify date range  | YEAR-MONTH-DAY  | 2020-01-01  | 2020-12-02  |
| -t  | To date: if one wants to specify date range  | YEAR-MONTH-DAY  | 2020-01-30  | 2020-12-20  |
| -l  | Test with limit: to speed up testing, samples only from data of this year/month/day  | YEARMONTHDAY  | 202001  | 20201220 |

NOTE: the **first** keyword entered is also used to prefix the data files and figures!


## Description of steps
1. Extract data
```bash
source /home/commando/maris/bin/activate
python extract_data.py $*
```
Extracts data from ```'/data/001_twitter_hope/preprocessed/da/*.ndjson'``` - this includes all preprocessed Danish Twitter data. Creates a file with matches with keywords per file and saves them to data folder.

2. Join files
```bash
python join_files.py $*
```
Joins together all files starting with keyword1 in the data folder, joins them together as *keyword1_data.csv*. Deletes all the keyword files in data folder to reduce taking up space.

3. Semantic scores
```bash
source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate
python semantic_scores.py $*
```
Calculates semantic scores with Danish Vader per tweet. Outputs *keyword1_data_SA.csv*

4. Preprocess stats
```bash
source /home/commando/maris/bin/activate
python preprocess_stats.py $*
```
Prepares the dataframe for visualizations, outputs statistics. These can be captures in the logs. Outputs *keyword1_vis.csv*

5. Visualize
```bash
python visualize.py $*
```
Creates visuals: keyword mentions frequency over time, compound sentiment over time, frequent hashtags, frequent words, wordcloud, bigram graphs with k varying between 1 and 5. Saves to *fig*