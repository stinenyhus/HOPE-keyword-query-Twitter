#!/bin/sh

source /home/commando/.virtualenvs/ss/bin/activate
python extract_data.py $*
python join_files.py $* 

python preprocess_stats.py $* 

source /home/stine/.virtualenvs/hope/bin/activate
python sentiment_bert.py $* 

source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate
python semantic_scores.py $* 

source /home/saram/.virtualenvs/hope/bin/activate
python smooth_and_entropy.py $*
python visualize.py $*
python streamlit_prep.py $*
