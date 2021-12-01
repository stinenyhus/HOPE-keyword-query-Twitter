#!/bin/sh

source /home/commando/.virtualenvs/ss/bin/activate
python extract_data.py $*
python join_files.py $* #output _data

python preprocess_stats.py $* #input _data, output _data_pre

source /home/commando/covid_19_rbkh/Preprocessing/text_to_x/bin/activate
python semantic_scores.py $* #input _data_pre, output _vis

source /home/commando/.virtualenvs/ss/bin/activate
python smooth_and_entropy.py $*
python visualize.py $*
