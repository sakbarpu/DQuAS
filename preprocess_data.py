import sys, os
import json
import pickle
import collections

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
dev_data_files = os.listdir(dev_dir)
print("There are ", len(train_data_files), "validation JSON files")
print("Listed below")
for f in dev_data_files: print(f)
print("")

def extract_qa(line_dict):
	question_text = line_dict['question_text'] + '?'

	for la in line_dict['annotations']:
		long_answer = la['long_answer']
		la_start_byte = long_answer['start_byte']
		la_end_byte = long_answer['end_byte']
		long_answer_content = ""
		for l in line_dict['document_tokens']:
			if l['start_byte'] >= la_start_byte and l['end_byte'] <= la_end_byte and l['html_token'] is False:
				 long_answer_content += l['token'] + " "
	
	lac_starts = collections.defaultdict(list)
	lac_ends = collections.defaultdict(list)
	i = 0
	for lac in random.sample(line_dict['long_answer_candidates'], min(5, len(line_dict['long_answer_candidates']))):
		if lac['start_byte'] == la_start_byte and lac['end_byte'] == la_end_byte:
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
	
	return [question_text, long_answer_content, lac_contents]

def preprocess_data(full_data):
	qa_ds = []
	c = 0
	for i, l in enumerate(full_data):
		if c%1000 == 0: print(c, '/', len(full_data))
		json_data = json.loads(l)
		qa = extract_qa(json_data)
		qa_ds.append(qa)
		c += 1
	return [a for a in qa_ds if len(a[1]) > 0]

print("\n\nNow preprocessing the training datasets...")
qa_dataset = []
num_train_files = 0
output_file = os.path.join(train_dir, "train_processed.csv")

with open(output_file, 'w') as csvfile:
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

with open(output_file) as csvfile:
	for dev_file in sorted(dev_data_files):
		dev_file = os.path.join(dev_dir, dev_file)
		print(num_dev_files, ": ", dev_file)

		with open(dev_file) as f:
			content = f.readlines()
			qa = preprocess_data(content)
			qa_dataset += qa
		del content
		f.close()
		num_dev_files += 1
		print("Number of examples so far: ", len(qa_dataset))

		for qa in qa_dataset:
			csvfile.write(qa[0] + " <;;;> " + " 1 " + " <;;;> " + qa[1] + "\n")
			for cand in qa[2]:
				csvfile.write(qa[0] + " <;;;> " + " 0 " + " <;;;> " + cand + "\n")


