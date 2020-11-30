import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

bert_model = 'albert-large-v2'
tokenizer = AutoTokenizer.from_pretrained(bert_model)
lens_ques = []
lens_cands = []
for index, row in tqdm(df_train.iterrows()):
	lens_ques.append(len(row['question']))
	cand_seq = tokenizer.encode(row['candidate'])
	lens_cands.append(len(cand_seq))

print(min(lens_ques), max(lens_ques), np.mean(lens_ques), np.std(lens_ques))
print(min(lens_cands), max(lens_cands), np.mean(lens_cands), np.std(lens_cands))

