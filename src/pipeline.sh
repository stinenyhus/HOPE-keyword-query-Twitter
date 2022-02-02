#!/bin/sh

source /home/commando/.virtualenvs/ss/bin/activate
python extract_data.py $*
python join_files.py $* #output _data

python preprocess_stats.py $* #input _data, output _data_pre

# source /home/commando/.virtualenvs/info/bin/activate
source /home/stine/.virtualenvs/hope/bin/activate
python sentiment_bert.py $* 

source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate
python semantic_scores.py $* #input _data_pre, output _vis

source /home/saram/.virtualenvs/hope/bin/activate
python smooth_and_entropy.py $*
python visualize.py $*
python streamlit_prep.py $*
