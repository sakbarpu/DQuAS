import sys, os
import json
import pickle
import collections
from tqdm import tqdm
from train import SentencePairClassifier
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):

	def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):
		self.data = data  # pandas dataframe
		#Initialize the tokenizer
		self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
		self.maxlen = maxlen
		self.with_labels = with_labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		# Selecting sentence1 and sentence2 at the specified index in the data frame
		exampleid = str(self.data.loc[index, 'example_id'])
		sent1 = str(self.data.loc[index, 'question'])
		sent2 = str(self.data.loc[index, 'candidate'])
		start = str(self.data.loc[index, 'start'])
		end = str(self.data.loc[index, 'end'])

		# Tokenize the pair of sentences to get token ids, attention masks and token type ids
		encoded_pair = self.tokenizer.encode_plus(sent1, sent2, pad_to_max_length=True,  # Pad to max_length
								#truncation=True,	# Truncate to max_length
								max_length=self.maxlen,
								return_tensors='pt')	# Return torch.Tensor objects

		token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
		attn_masks = encoded_pair['attention_mask'].squeeze(0)	# binary tensor with "0" for padded values and "1" for the other values
		token_type_ids = encoded_pair['token_type_ids'].squeeze(0)	# binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

		if self.with_labels:  # True if the dataset has labels
			label = self.data.loc[index, 'label']
			return exampleid, token_ids, attn_masks, token_type_ids, start, end, label
		else:
			return exampleid, token_ids, attn_masks, token_type_ids, start, end

def get_probs_from_logits(logits):
	"""
	Converts a tensor of logits into an array of probabilities by applying the sigmoid function
	"""
	probs = torch.sigmoid(logits.unsqueeze(-1))
	return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
	"""
	Predict the probabilities on a dataset with or without labels and print the result in a file
	  {'predictions': [
	    {

	      'example_id': -2226525965842375672,
	      'long_answer': {
		'start_byte': 62657, 'end_byte': 64776,
		'start_token': 391, 'end_token': 604
	      },
	      'long_answer_score': 13.5,

	      'short_answers': [
		{'start_byte': 64206, 'end_byte': 64280,
		 'start_token': 555, 'end_token': 560}, ...],
	      'short_answers_score': 26.4,

	      'yes_no_answer': 'NONE'

	    }, ... ]

	  }
	"""

	net.eval()
	data = {}
	data['predictions'] = []
	probs_all = collections.defaultdict(float)
	negone = -1
	with torch.no_grad():
		for eids, seq, attn_masks, token_type_ids, starts, ends, _ in tqdm(dataloader):
			seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
			logits = net(seq, attn_masks, token_type_ids)
			probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
			#probs = [1.0 for i in range(len(eids))]
			for eid, start, end, prob in zip(eids, starts, ends, probs):
				if prob > probs_all[eid]:
					probs_all[eid] = prob
					data['predictions'].append({
						"example_id": int(eid),
						"long_answer": {
							'start_byte': int(start), 'end_byte': int(end),
							'start_token': -1, 'end_token': -1
						},
						#"long_answer_score": prob.item(),
						"long_answer_score": prob,
						
						"short_answer": [
						],
						"short_answers_score": -1,
						'yes_no_answer': 'NONE'
					})

	with open(result_file, 'w') as outfile:
		json.dump(data, outfile)


def extract_qa(line_dict):
	question_text = line_dict['question_text'] + '?'
	example_id = line_dict['example_id']
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
		long_answers.append([long_answer_content, la_start_byte, la_end_byte])


	lac_starts = collections.defaultdict(list)
	lac_ends = collections.defaultdict(list)
	i = 0
	lac_contents = []
	for lac in line_dict['long_answer_candidates']:
		if lac['start_byte'] in la_start_bytes_set and lac['end_byte'] in la_end_bytes_set:
			continue
		start_byte = lac['start_byte']
		end_byte = lac['end_byte']
		lac_starts[start_byte].append(i)
		lac_ends[end_byte].append(i)
		i += 1
		lac_contents.append(["", start_byte, end_byte])

	curr_spans = set()
	for l in line_dict['document_tokens']:

		if l['start_byte'] in lac_starts:
			for x in lac_starts[l['start_byte']]:
				curr_spans.add(x)

		if l['end_byte'] in lac_ends:
			for x in lac_ends[l['end_byte']]:
				curr_spans.remove(x)

		for span in curr_spans:
			lac_contents[span][0] += l['token'] + " "

	return [example_id, question_text, long_answers, lac_contents]

def preprocess_data(full_data):
	qa_ds = []
	print(len(full_data))
	for i, l in tqdm(enumerate(full_data)):
		json_data = json.loads(l)
		qa = extract_qa(json_data)
		if qa: qa_ds.append(qa)
	return [a for a in qa_ds if len(a[1]) > 0]

data_dir = sys.argv[1]
dev_dir = os.path.join(data_dir,"whole_data","dev")
dev_data_files = [x for x in os.listdir(dev_dir) if x.endswith("jsonl")]
print("There are ", len(dev_data_files), "validation JSON files")
print("Listed below")
for f in dev_data_files: print(f)
print("")

print("\n\nNow preprocessing the dev datasets...")
dev_path_new = os.path.join(sys.argv[1],"whole_data", 'dev/dev_processed_new_0.csv')
if not True:
	qa_dataset = []
	num_dev_files = 0
	with open(dev_path_new, "w") as csvfile:
		csvfile.write("example_id" + " <;;;> " + "question" + " <;;;> " + "label" + " <;;;> " + "candidate" + " <;;;> " + "start" + " <;;;> " + "end\n")
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
			num_devs = 0
			for qa in qa_dataset:
				for la in qa[2]:
					csvfile.write(str(qa[0]) + " <;;;> " + qa[1] + " <;;;> " + " 1 " + " <;;;> " + la[0] + " <;;;> " +  str(la[1]) + " <;;;> " + str(la[2]) + "\n")
					num_devs += 1
				for cand in qa[3]:
					csvfile.write(str(qa[0]) + " <;;;> " + qa[1] + " <;;;> " + " 0 " + " <;;;> " + cand[0] + " <;;;> " +  str(cand[1]) + " <;;;> " + str(cand[2]) + "\n")
					num_devs += 1

			print("Number of dev examples so far: ", num_devs)
			break
			
	del qa_dataset
	csvfile.close()

#/home/ubuntu/mnt/cloudNAS2/SoftKBase/Google_NQ/models/bert-base-uncased_lr_1e-06_val_loss_10.62006_ep_0_1.pt
path_to_model = sys.argv[1] + 'models/bert-base-uncased_lr_1e-06_val_loss_10.62006_ep_0_1.pt'	
path_to_output_file = sys.argv[1] + 'results/output.json'

print("Reading test data...")
bert_model = 'bert-base-uncased'
delimiter = " <;;;> "
maxlen = 512
df_test = pd.read_csv(dev_path_new, delimiter=delimiter)
test_set = CustomDataset(df_test, maxlen, bert_model)
test_loader = DataLoader(test_set, batch_size=32, num_workers=1)

model = SentencePairClassifier(bert_model)
print()

print("Loading the weights of the model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(path_to_model, map_location=device))
#model.to(device)

print("Predicting on test data...")
test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
				result_file=path_to_output_file)
#print()
#print("Predictions are available in : {}".format(path_to_output_file))
#
#path_to_output_file = 'results/output.txt'  # path to the file with prediction probabilities
#labels_test = df_test['label']  # true labels
#
#probs_test = pd.read_csv(path_to_output_file, header=None)[0]	# prediction probabilities
#threshold = 0.5   # you can adjust this threshold for your own dataset
#preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold
#
#metric = load_metric("glue", "mrpc")
## Compute the accuracy and F1 scores
#metric._compute(predictions=preds_test, references=labels_test)

