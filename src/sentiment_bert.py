'''
 Conducting sentiment/emotion analysis uding BERT
'''

import os
import pandas as pd
import spacy
from csv import writer 
import getopt, sys
import re
import ssl
from dacy.sentiment import add_bertemotion_emo, add_bertemotion_laden, add_berttone_polarity
import pysentimiento as ps
from configparser import ConfigParser



# Trust the DaNLP site 
ssl._create_default_https_context = ssl._create_unverified_context

### define functions ###
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def bert_scores(data_prefix: str, out_path:str):
    filename = os.path.join("..", f"{data_prefix}_files", f"{data_prefix}_data_pre.csv")
    df = pd.read_csv(filename)

    # Prepare BERT models
    print('Prepare models')
    nlp = spacy.blank("da")
    # Doc.set_extension("vader_da", getter=da_vader_getter, force = True) #da_vader_getter is from dacy
    nlp = add_bertemotion_laden(nlp)   
    nlp = add_bertemotion_emo(nlp) 
    nlp = add_berttone_polarity(nlp)#, force_extension=True) 

    old_cols = list(df.columns[1:])
    out = pd.DataFrame(columns = old_cols + [ # existing columns 
                                "emotional",
                                "emo_score",
                                "emotional_prob",
                                "emotion",
                                "emotion_prob", 
                                "polarity", 
                                "polarity_score",
                                "polarity_prob"])
    out.to_csv(f'{out_path}.csv')

    # Ensure that all texts are strings to avoid float error (hopefully)
    df["mentioneless_text"] = [str(text) for text in df["mentioneless_text"]]
    docs = nlp.pipe(df["mentioneless_text"])

    emo_dict = {"Emotional": 1,
                "No emotion": 0}
    pol_dict = {"positive": 1,
                "neutral": 0,
                "negative": -1}

    for idx, doc in enumerate(docs):
        # BERT emotion
        emotional = doc._.laden
        emotional_prob = doc._.laden_prop["prop"]
        emotion = doc._.emotion
        emotion_prob = doc._.emotion_prop["prop"]

        # BERT polarity 
        pol_label = doc._.polarity
        pol_label_prob = doc._.polarity_prop["prop"]

        row = [idx]+[df[column][idx] for column in old_cols]
        row += [emotional,
                emo_dict[emotional],
                emotional_prob,
                emotion,
                emotion_prob,
                pol_label,
                pol_dict[pol_label],
                pol_label_prob]
        append_list_as_row(f'{out_path}.csv', row)

def english_sentiment(column:pd.Series):
    """This function extracts Pysentimentio sentiment.
    Object called analyzer, a pysentimentio analyzer object, must be created prior to use
    Returns a tuple with the predicted label and the associated probability

    Args:
        column (pd.Series): column in dataframe to apply sentiment to

    Returns:
        tuple: (polarity label, polarity prob) or ("nan", "nan") for "nan" input
    """    
    if pd.isna(column):
        return (float("nan"), float("nan"))
    output = analyzer.predict(column)
    return (output.output, output.probas[output.output])


def bert_scores_en(data_prefix: str, out_path:str):
    # Prepare file
    filename=os.path.join("..", f"{data_prefix}_files", f"{data_prefix}_data_pre.csv")
    df = pd.read_csv(filename,lineterminator='\n')

    # Apply using analyzer
    analyzer = ps.SentimentAnalyzer(lang="en")
    sentiment = list(map(english_sentiment, df["mentioneless_text"]))
    print("Calculated sentiment")
    sents = list(zip(*sentiment))

    # Dictionary for getting polarity score
    pol_dict = {"POS": 1,
                "NEU": 0,
                "NEG": -1}

    # Add to df
    df["polarity"] = sents[0]
    df["polarity_score"] = [pol_dict[label] if not pd.isna(label) else label for label in df["polarity"]]
    df["polarity_prob"] = sents[1]

    df.to_csv(f'{out_path}.csv')

def main(argv):
    keywords = ''
    from_date = '' 
    to_date = ''
    test_limit = '' # this is so that it's possible to test the system on just one day/month of data
    small = ''
    try:
        opts, args = getopt.getopt(argv,"hk:")
        config = ConfigParser()
        config.read("keyword_config.ini")
        
    except getopt.GetoptError:
        print('test.py -k <keyword1,keyword2> -f <2020-01-20> -t <2020-12-31> -l <20200101>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -keywords <keyword1,keyword2> -from_date <2020-01-20> -to_date <2020-12-31> -test_limit <20200101>')
            sys.exit()
        elif opt in "-k":
            key = arg
            keywords = config[f'{key}']["keywords"]
            from_date = config[f'{key}']["from_date"]
            to_date = config[f'{key}']["to_date"]
            test_limit = config[f'{key}']["test_limit"]
            small = config[f'{key}']["small"]
            language = config[f'{key}']["lan"]
            print(f'Running BERT models with key: {key}, keywords: {keywords} from {from_date} and small = {small}')
    return keywords, language


if __name__ == "__main__":
    keywords, language = main(sys.argv[1:])
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
    out = os.path.join("..", f'{data_prefix}_files', f'{data_prefix}_data_bert')
    if language == "en":
        bert_scores_en(data_prefix, out)
    else:    
        bert_scores(data_prefix, out)