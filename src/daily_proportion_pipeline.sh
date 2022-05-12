source /home/commando/.virtualenvs/ss/bin/activate
python extract_data.py $*
python join_files.py $* 

python preprocess_stats.py $* 

python all_covid_ids.py $*
python daily_proportion.py $*