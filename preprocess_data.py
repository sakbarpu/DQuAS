import sys, os
import json
import pickle
import collections
import random
from math import ceil
import heapq
from tqdm import tqdm

data_dir = sys.argv[1]

print("The input data dir is: ", data_dir)
print(os.listdir(data_dir))

train_dir = os.path.join(data_dir,"train")
train_data_files = [x for x in os.listdir(train_dir) if x.endswith("jsonl")]
print("There are ", len(train_data_files), "training JSON files")
print("Listed below")
for f in train_data_files: print(f)
print("")

dev_dir = os.path.join(data_dir,"dev")
dev_data_files = [x for x in os.listdir(dev_dir) if x.endswith("jsonl")]
print("There are ", len(dev_data_files), "validation JSON files")
print("Listed below")
for f in dev_data_files: print(f)
print("")

def find_closest_len_candidate_ids(cands, length, k):
	return heapq.nsmallest(k, cands, key=lambda x: abs(x[0]-length))

def find_smallest_len_candidate_ids(cands, k):
	return heapq.nsmallest(k, cands)

def find_largest_len_candidate_ids(cands, k):
	return heapq.nlargest(k, cands)
 
def extract_qa(line_dict, training=True):
	question_text = line_dict['question_text'] + '?'
	if training:
		long_answer = line_dict['annotations'][0]['long_answer']
		la_start_byte = long_answer['start_byte']
		la_end_byte = long_answer['end_byte']
		if la_start_byte < 0 and la_end_byte < 0: 
			return

		long_answer_content = ""
		for l in line_dict['document_tokens']:
			if l['start_byte'] >= la_start_byte and l['end_byte'] <= la_end_byte and l['html_token'] is False:
				 long_answer_content += l['token'] + " "
		long_answer_len = la_end_byte - la_start_byte + 1

		candidates_start_end_len = [( lac['end_byte']-lac['start_byte']+1, lac['start_byte'], lac['end_byte'] ) for lac in line_dict['long_answer_candidates']]
		heapq.heapify(candidates_start_end_len)

		num_of_candidates = ceil(0.10*len(line_dict['long_answer_candidates']))
		closest_cands = find_closest_len_candidate_ids(candidates_start_end_len, long_answer_len, ceil(0.5 * num_of_candidates))
		smallest_cands = find_smallest_len_candidate_ids(candidates_start_end_len, ceil(0.25 * num_of_candidates))
		largest_cands = find_largest_len_candidate_ids(candidates_start_end_len, ceil(0.25 * num_of_candidates))
		all_cands = closest_cands + smallest_cands + largest_cands
		all_cands = list(set(all_cands))

		lac_starts = collections.defaultdict(list)
		lac_ends = collections.defaultdict(list)
		i = 0
		for lac in all_cands:
			start_byte = lac[1]
			end_byte = lac[2]
			if start_byte == la_start_byte and end_byte == la_end_byte:
				continue
			lac_starts[start_byte].append(i)
			lac_ends[end_byte].append(i)
			i += 1
		
		num_of_candiates = len(all_cands)
		lac_contents = ["" for x in range(i)]
		curr_spans = set()
		for l in line_dict['document_tokens']:

			if l['start_byte'] in lac_starts:
				for x in lac_starts[l['start_byte']]:
					curr_spans.add(x)

			if l['end_byte'] in lac_ends:
				for x in lac_ends[l['end_byte']]:
					curr_spans.remove(x)

			for span in curr_spans:
				lac_contents[span] += l['token'] + " "
		return [question_text, long_answer_content, lac_contents]

	else:
		long_answers = []
		la_start_bytes_set = set()
		la_end_bytes_set = set()
		for la in line_dict['annotations']:
			long_answer = la['long_answer']
			la_start_byte = long_answer['start_byte']
			la_end_byte = long_answer['end_byte']
			if la_start_byte < 0 and la_end_byte < 0: continue

			la_start_bytes_set.add(la_start_byte)
			la_end_bytes_set.add(la_end_byte)
			long_answer_content = ""
			for l in line_dict['document_tokens']:
				if l['start_byte'] >= la_start_byte and l['end_byte'] <= la_end_byte and l['html_token'] is False:
					 long_answer_content += l['token'] + " "
			long_answers.append(long_answer_content)	

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
				lac_contents[span] += l['token'] + " "
		
		return [question_text, long_answers, lac_contents]

def preprocess_data(full_data, training=True):
	qa_ds = []
	print(len(full_data))
	for i, l in tqdm(enumerate(full_data)):
		json_data = json.loads(l)
		qa = extract_qa(json_data, training)
		if qa: qa_ds.append(qa)
	return [a for a in qa_ds if len(a[1]) > 0]

print("\n\nNow preprocessing the training datasets...")
qa_dataset = []
num_train_files = 0
output_file = os.path.join(train_dir, "train_processed.csv")
with open(output_file, 'w') as csvfile:
	csvfile.write("question" + " <;;;> " + "label" + " <;;;> " + "candidate\n")
	for train_file in sorted(train_data_files):
		train_file = os.path.join(train_dir, train_file)
		print(num_train_files, ": ", train_file)

		with open(train_file) as f:
			content = f.readlines()
			qa = preprocess_data(content)
			qa_dataset += qa
		del content
		f.close()
		num_train_files += 1
		print("Number of examples so far: ", len(qa_dataset))

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

		with open(dev_file) as f:
			content = f.readlines()
			qa = preprocess_data(content, False)
			qa_dataset += qa
		del content
		f.close()
		num_dev_files += 1
		print("Number of examples so far: ", len(qa_dataset))

		for qa in qa_dataset:
			for la in qa[1]:
				csvfile.write(qa[0] + " <;;;> " + " 1 " + " <;;;> " + la + "\n")
			for cand in qa[2]:
				csvfile.write(qa[0] + " <;;;> " + " 0 " + " <;;;> " + cand + "\n")

del qa_dataset
csvfile.close()
