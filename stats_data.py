'''
The main use for this script is just to explore the dataset
to see how big the length of questions and candidates are.

I have reported the results in stat.txt file.

Here is how to use this script:

	python stats_data.py <path_data>

Where path_data is the path where processed data is stored.
That is, where the output of preprocess_data.py is stored.
For example, path to train_processed.csv
'''

import sys, os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

train_path = sys.argv[1]
delimiter = " <;;;> "
df_train = pd.read_csv(train_path, delimiter=delimiter)

print("Number of examples", print(len(df_train)))
print("Number of 1 labels", print(df_train['label'].value_counts()))

bert_model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_model)
lens_ques = []
lens_cands = []
for index, row in tqdm(df_train.iterrows()):
	lens_ques.append(len(row['question']))
	cand_seq = tokenizer.encode(row['candidate'])
	lens_cands.append(len(cand_seq))

print(min(lens_ques), max(lens_ques), np.mean(lens_ques), np.std(lens_ques))
print(min(lens_cands), max(lens_cands), np.mean(lens_cands), np.std(lens_cands))

