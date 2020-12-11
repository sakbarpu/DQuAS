'''
This script performs an exploratory study of the dataset.

Here is how to use it:

	python explore_dataset.py <data_dir>

where data_dir is the dir where data is stored.

'''

import sys, os
import json

data_dir = sys.argv[1]

print("The input data dir is: ", data_dir)
print(os.listdir(data_dir))

train_dir = os.path.join(data_dir,"train")
train_data_files = os.listdir(train_dir)
print("There are ", len(train_data_files), "training JSON files")
print("Listed below")
for f in train_data_files: print(f)
print("")

dev_dir = os.path.join(data_dir,"dev")
dev_data_files = os.listdir(dev_dir)
print("There are ", len(train_data_files), "validation JSON files")
print("Listed below")
for f in dev_data_files: print(f)
print("")

print("Train data size: ")
os.system("du -hs " + train_dir)
print("Dev data size: ")
os.system("du -hs " + dev_dir)
print("")

print("A sample train file has this many number of lines:")
os.system("wc -l " + os.path.join(train_dir,train_data_files[0]))
print("A sample dev file has this many number of lines:")
os.system("wc -l " + os.path.join(dev_dir,dev_data_files[0]))
print("")

print("Let's open a training JSON file and see what is in there.")
with open(os.path.join(train_dir, train_data_files[0])) as f:
	content = json.loads(f.readline())
	print(content.keys())
	# returns dict_keys(['annotations', 'document_html', 'document_title', 
	#		     'document_tokens', 'document_url', 'example_id', 
	#		     'long_answer_candidates', 'question_text', 
	#		     'question_tokens'])
	print("")

	print("Question Text:")
	print(content['question_text'])	
	print("")

	print("Question Tokens:")
	print(content['question_tokens'])
	print("")

	print("Document URL")
	print(content['document_url'])
	print("")
	
	print("Document Title:")
	print(content['document_title'])
	print("")
	
	print("Document Tokens Len:")
	print(len(content['document_tokens']))
	print("")

	print("Document Tokens First One:")
	print(content['document_tokens'][0])
	print("")

	print("Long Answer Candidates Len:")
	print(len(content['long_answer_candidates']))
	print("")

	print("Long Answer Candidates First One:")
	print(content['long_answer_candidates'][0])
	print("")

	print("Annotations:")
	print(content['annotations'])
	print("")
