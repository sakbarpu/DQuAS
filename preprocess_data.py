'''
This script preprocesses the dataset and stores them in a big file.
Specifically, we are interested in the training dataset.

The dev dataset is correctly processed in the prediciton.py script.
By correct I mean the way it should be processed in order to conform
to the specifications of nq_eval.py file. (More info in prediction.py file)

The way to use this script is as follows:

	python preprocess_data.py <data_dir>

where data_dir is the root location for the whole data. By whole
I mean both training and dev data.

data_dir
	|_ train
	|_ dev

After this script runs there are two files created and stored in 
train and dev dirs.

data_dir
	|_ train
		|_ train_processed.csv
	|_ dev
		|_ dev_processed.csv

The content of train_processed.csv is comma seperated three things:
(1) question text (2) label either 0 or 1 (3) candidate

Obviously for actual positive examples label is 1, and 0 otherwise.
An example from train_processed.csv

	question <;;;> label <;;;> candidate
	when is the last episode of season 8 of the walking dead? <;;;>  0  <;;;> <Li> Mandi Christine Kerr as Barbara , resident of Alexandria .

Notice the delimiter is very special sequence of chars <;;;>
I just came up with that so that I can distinguish it from anything 
else that might appear in the text.
'''


import sys, os
import json
import pickle
import collections
import random
from math import ceil
import heapq
from tqdm import tqdm

def find_closest_len_candidate_ids(cands, length, k):
	'''
	Get the k items from heap cands that have their lengths
	closest to the length parameter.

	:type cands: List
	:type length: int
	:type k: int
	:rtype List
	'''

	return heapq.nsmallest(k, cands, key=lambda x: abs(x[0]-length))

def find_smallest_len_candidate_ids(cands, k):
	'''
	Get the k smallest items from heap cands

	:type cands: List 
	:type k: int
	:rtype List
	'''

	return heapq.nsmallest(k, cands)

def find_largest_len_candidate_ids(cands, k):
	'''
	Get the k largest items from heap cands

	:type cands: List
	:type k: int
	:rtype List
	'''

	return heapq.nlargest(k, cands)

def extract_train_qa(line_dict):
	'''
	For a particular example in the training set it returns three things.
	(1) the question text that can be extracted easily from question_text field

	(2) the correct long answer (i.e. positive label) 
		we get the start_byte and end_byte from annotations -> long_answer field
		then we loop over the entire document tokens and
		gather those tokens that are within range [start_byte, end_byte]
		and form the long_answer_content

	(3) the negative samples from long_answer_candidates list
		we sample 10% of the negative examples from the list
		out of these 10%, 50% are candidates with length similar to actual positive long answer
		and 25% are smallest candidates and 25% are largest candidates
		we maintain a heap of tuples (length, start_byte, end_byte) to organize the candidates
		and for efficiently retrieve our desired candidates

		after that we have obtained the (len, start, end) tuples we store start bytes in lac_starts dict
		and end bytes in lac_ends dict. 
		and for each key start_byte we store a list of candidates that start with the key start_byte
		and for each key end_byte we store a list of candidates that end with the key end_byte
	
		we then start a list of long answer candidates that we want to store with empty strings

		after that we loop over all the tokens and see if we are in the span of any of the candidates
		if we are then we add the document token to the appropriate candidate in the list of candidates
	
	:type line_dict: Dict
	:rtype List
	'''

	# Get the question text (nice and simple)
	question_text = line_dict['question_text'] + '?'

	# Get the long answer content (positive examples)
	long_answer = line_dict['annotations'][0]['long_answer']
	la_start_byte = long_answer['start_byte']
	la_end_byte = long_answer['end_byte']
	# We ignore this example if there is no long answer 
	if la_start_byte < 0 and la_end_byte < 0: 
		return
	# Now loop over doc tokens and gather long answer
	long_answer_content = ""
	for l in line_dict['document_tokens']:
		# is this token within range of long asnwer span (start to end byte) and also not a regular html token
		if l['start_byte'] >= la_start_byte and l['end_byte'] <= la_end_byte and l['html_token'] is False:
			 long_answer_content += l['token'] + " "
	long_answer_len = la_end_byte - la_start_byte + 1

	# Get the candidates (negative examples)
	# Get list of tuples (len, start, end) and heapify based on len
	candidates_start_end_len = [( lac['end_byte']-lac['start_byte']+1, lac['start_byte'], lac['end_byte'] ) for lac in line_dict['long_answer_candidates']]
	heapq.heapify(candidates_start_end_len)
	# Now sample 10% of the examples based on the criteria described above (closest + smallest + largest)
	num_of_candidates = ceil(0.10*len(line_dict['long_answer_candidates']))
	closest_cands = find_closest_len_candidate_ids(candidates_start_end_len, long_answer_len, ceil(0.5 * num_of_candidates))
	smallest_cands = find_smallest_len_candidate_ids(candidates_start_end_len, ceil(0.25 * num_of_candidates))
	largest_cands = find_largest_len_candidate_ids(candidates_start_end_len, ceil(0.25 * num_of_candidates))
	all_cands = closest_cands + smallest_cands + largest_cands
	all_cands = list(set(all_cands))
	# Now create dict of starts and ends bytes
	lac_starts = collections.defaultdict(list)
	lac_ends = collections.defaultdict(list)
	i = 0
	for lac in all_cands:
		start_byte = lac[1]
		end_byte = lac[2]
		# Make sure to ignore the positive example (long answer content) from above
		# We don't want to put them as negative label obviously
		if start_byte == la_start_byte and end_byte == la_end_byte:
			continue
		lac_starts[start_byte].append(i)
		lac_ends[end_byte].append(i)
		i += 1
	# Init a list of candidate contents as empty string
	# Simply loop over doc tokens and populate strings
	num_of_candiates = len(all_cands)
	lac_contents = ["" for x in range(i)]
	curr_spans = set() #this stores all the candidates that are in current span
	for l in line_dict['document_tokens']:
		# Is this start byte seen in starts dict
		if l['start_byte'] in lac_starts:
			# If so include all the candidates in the current span
			for x in lac_starts[l['start_byte']]:
				curr_spans.add(x)
		# Is this end byte seen in ends dict
		if l['end_byte'] in lac_ends:
			# If so remove all the candidates from the current span
			for x in lac_ends[l['end_byte']]:
				curr_spans.remove(x)
		# Obviously every candidate that is present in curr span should include this token in its content
		for span in curr_spans:
			if l['html_token'] is False:
				lac_contents[span] += l['token'] + " "

	return [question_text, long_answer_content, lac_contents]

def extract_dev_qa(line_dict):
	'''
	For each dev entry line_dict in the dev dataset we create the following:
	(1) question text
	(2) list of all long answers that were annotated (total 5 answers)
	(3) list of all candidates that are present for this example
	
	The way this function works is very similar to what happens in extract_train_qa()

	:type line_dict: Dict
	:rtype List
	'''

	# Nice and simple extract the question text
	question_text = line_dict['question_text'] + '?'

	# Get the 5 long answers annotated in the dataset
	long_answers = []
	la_start_bytes_set = set() #start bytes of answers for later usage when we process candidates and want to ignore pos examples
	la_end_bytes_set = set() #end bytes of answers for later usage when we process candidates and want to ignore pos examples
	# Loop over all the 5 annotations that contain long answer
	for la in line_dict['annotations']: 
		# Get this curr long answer start and end bytes
		long_answer = la['long_answer']
		la_start_byte = long_answer['start_byte']
		la_end_byte = long_answer['end_byte']
		# Obviously we ignore if there is no long answer there
		if la_start_byte < 0 and la_end_byte < 0: 
			continue
		# Add start and end bytes to sets
		la_start_bytes_set.add(la_start_byte)
		la_end_bytes_set.add(la_end_byte)
		# Update the long answer content of this answer
		# Loop over doc tokens and gather only tokens when they are in span of this answer
		long_answer_content = ""
		for l in line_dict['document_tokens']:
			# Is this token in the span of the current answer and also not a html token
			if l['start_byte'] >= la_start_byte and l['end_byte'] <= la_end_byte and l['html_token'] is False:
				 long_answer_content += l['token'] + " "
		long_answers.append(long_answer_content)	

	# Get the long answer candidates
	# See the prediction.py file for correct way to do this
	lac_starts = collections.defaultdict(list)
	lac_ends = collections.defaultdict(list)
	i = 0
	for lac in random.sample(line_dict['long_answer_candidates'], len(line_dict['long_answer_candidates'])):
		if lac['start_byte'] in la_start_bytes_set and lac['end_byte'] in la_end_bytes_set:
			continue
		start_byte = lac['start_byte']
		end_byte = lac['end_byte']
		lac_starts[start_byte].append(i)
		lac_ends[end_byte].append(i)
		i += 1

	lac_contents = ["" for x in range(len(lac_starts))]
	curr_spans = set()
	for l in line_dict['document_tokens']:

		if l['start_byte'] in lac_starts:
			for x in lac_starts[l['start_byte']]:
				curr_spans.add(x)

		if l['end_byte'] in lac_ends:
			for x in lac_ends[l['end_byte']]:
				curr_spans.remove(x)

		for span in curr_spans:
			if l['html_token'] is False:
				lac_contents[span] += l['token'] + " "
	
	return [question_text, long_answers, lac_contents]


def extract_qa(line_dict, training=True):
	'''
	Either run it for training data or for dev data.
	Both treated differently for obvious reasons.

	:type line_dict: Dict
	:type training: bool
	:rtype 
	'''

	if training:
		return extract_train_qa(line_dict)
	else:
		return extract_dev_qa(line_dict)

def preprocess_data(full_data, training=True):
	'''
	Takes as input data read from the file and creates
	a json object and pass to extract_qa() for further
	processing of each individual data item. Finally,
	forms a list of Question Answer data and returns.

	:type full_data: List
	:type training: bool
	:rtype List
	'''
	qa_ds = []
	print("Length of data:", len(full_data))
	for i, l in tqdm(enumerate(full_data)):
		json_data = json.loads(l)
		qa = extract_qa(json_data, training)
		if qa: qa_ds.append(qa)
	return [a for a in qa_ds if len(a[1]) > 0]

def main(data_dir):
	'''
	Just take as input data dir and store the training and 
	dev processed files at appropriate locations
	<data_dir>/train/train_processed.csv
	<data_dir>/dev/dev_processed.csv

	:type data_dir: str
	'''

	print("The input data dir is: ", data_dir)
	print(os.listdir(data_dir))

	# Get all the train files
	train_dir = os.path.join(data_dir,"train")
	train_data_files = [x for x in os.listdir(train_dir) if x.endswith("jsonl")]
	print("There are ", len(train_data_files), "training JSON files")
	print("Listed below")
	for f in train_data_files: print(f)
	print("")

	# Get all the dev files
	dev_dir = os.path.join(data_dir,"dev")
	dev_data_files = [x for x in os.listdir(dev_dir) if x.endswith("jsonl")]
	print("There are ", len(dev_data_files), "validation JSON files")
	print("Listed below")
	for f in dev_data_files: print(f)
	print("")

	print("\n\nNow preprocessing the training datasets...")
	qa_dataset = []
	num_train_files = 0
	output_file = os.path.join(train_dir, "train_processed.csv") #path to store preprocessed training data
	with open(output_file, 'w') as csvfile:
		csvfile.write("question" + " <;;;> " + "label" + " <;;;> " + "candidate\n")
		for train_file in sorted(train_data_files):
			train_file = os.path.join(train_dir, train_file)
			print(num_train_files, ": ", train_file)
			# Process this trian file
			with open(train_file) as f:
				content = f.readlines()
				qa = preprocess_data(content)
				qa_dataset += qa
			del content # For clearing memory
			f.close() 
			num_train_files += 1
			print("Number of examples so far: ", len(qa_dataset))
			# Write the content to the output CSV file
			for qa in qa_dataset:
				csvfile.write(qa[0] + " <;;;> " + " 1 " + " <;;;> " + qa[1] + "\n")
				for cand in qa[2]:
					csvfile.write(qa[0] + " <;;;> " + " 0 " + " <;;;> " + cand + "\n")

	del qa_dataset
	csvfile.close()

	print("\n\nNow preprocessing the dev datasets...")
	qa_dataset = []
	num_dev_files = 0
	output_file = os.path.join(dev_dir, "dev_processed.csv")
	with open(output_file, "w") as csvfile:
		csvfile.write("question" + " <;;;> " + "label" + " <;;;> " + "candidate\n")
		for dev_file in sorted(dev_data_files):
			dev_file = os.path.join(dev_dir, dev_file)
			print(num_dev_files, ": ", dev_file)
			# Process this dev file
			with open(dev_file) as f:
				content = f.readlines()
				qa = preprocess_data(content, False)
				qa_dataset += qa
			del content # For clearning memory
			f.close()
			num_dev_files += 1
			print("Number of examples so far: ", len(qa_dataset))
			# Write the content to the output CSV file
			for qa in qa_dataset:
				for la in qa[1]:
					csvfile.write(qa[0] + " <;;;> " + " 1 " + " <;;;> " + la + "\n")
				for cand in qa[2]:
					csvfile.write(qa[0] + " <;;;> " + " 0 " + " <;;;> " + cand + "\n")

	del qa_dataset
	csvfile.close()

if __name__ == "__main__":
	main(sys.argv[1])
