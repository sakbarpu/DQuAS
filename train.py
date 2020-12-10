
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
#from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DistilBertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
		sent1 = str(self.data.loc[index, 'question'])
		sent2 = str(self.data.loc[index, 'candidate'])

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
			return token_ids, attn_masks, token_type_ids, label  
		else:
			return token_ids, attn_masks, token_type_ids



class SentencePairClassifier(nn.Module):

	def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
		super(SentencePairClassifier, self).__init__()

		# Instantiating BERT-based model object
		#self.bert_layer = AutoModel.from_pretrained(bert_model)

		#  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
		if bert_model == "albert-base-v2":	# 12M parameters
			self.bert_layer = AutoModel.from_pretrained(bert_model)
			hidden_size = 768
		elif bert_model == "albert-large-v2":  # 18M parameters
			self.bert_layer = AutoModel.from_pretrained(bert_model)
			hidden_size = 1024
		elif bert_model == "albert-xlarge-v2":	# 60M parameters
			self.bert_layer = AutoModel.from_pretrained(bert_model)
			hidden_size = 2048
		elif bert_model == "albert-xxlarge-v2":  # 235M parameters
			self.bert_layer = AutoModel.from_pretrained(bert_model)
			hidden_size = 4096
		elif bert_model == "bert-base-uncased": # 110M parameters
			self.bert_layer = AutoModel.from_pretrained(bert_model)
			hidden_size = 768
		elif bert_model == "distilbert-base-uncased": #66M parameteres
			self.bert_layer = DistilBertModel.from_pretrained(bert_model)
			hidden_size = 768
		elif bert_model == 'distilbert-base-uncased-distilled-squad': #66M parameters
			self.bert_layer = DistilBertModel.from_pretrained(bert_model)
			hidden_size = 768

		# Freeze bert layers and only train the classification layer weights
		#if freeze_bert:
		#	for p in self.bert_layer.parameters():
		#		p.requires_grad = False

		# Classification layer
		self.cls_layer1 = nn.Linear(hidden_size, 1)
		self.dropout = nn.Dropout(p=0.1)

	#@autocast()  # run in mixed precision
	def forward(self, input_ids, attn_masks, token_type_ids):
		'''
		Inputs:
			-input_ids : Tensor  containing token ids
			-attn_masks : Tensor containing attention masks to be used to focus on non-padded values
			-token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
		'''

		# Feeding the inputs to the BERT-based model to obtain contextualized representations
		inputs = {"input_ids": input_ids, "attention_mask": attn_masks, "token_type_ids": token_type_ids,}
		c, pooler_output = self.bert_layer(**inputs)  

		#inputs = {"input_ids": input_ids, "attention_mask": attn_masks,}
		#output = self.bert_layer(**inputs)  

		# Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
		# Linear Layer and a Tanh activation. 
		# The Linear layer weights were trained from the sentence order prediction (ALBERT) 
		# or next sentence prediction (BERT)
		# objective during pre-training.
		#list_out = []
		#for i in range(len(output[0])):	
		#	list_out.append(output[0][i][0].unsqueeze(0))
		#pooler_output = torch.cat(list_out, dim=0)

		logits = self.cls_layer1(self.dropout(pooler_output))
		return logits

def set_seed(seed):
	""" Set all seeds to make results reproducible """
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_loss(net, device, criterion, dataloader):
	net.eval()
	mean_loss = 0
	count = 0
	out_labels = []
	out_preds = []
	with torch.no_grad():
		for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
			seq, attn_masks, token_type_ids, labels = \
				seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
			logits = net(seq, attn_masks, token_type_ids)
			preds = F.sigmoid(logits)
			out_labels.append(labels)
			out_preds.append(preds)
			mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
			count += 1

	all_preds = torch.cat(out_preds, dim=0).squeeze(-1)
	#print(all_preds>0.5)
	all_labels = torch.cat(out_labels, dim=0)
	#print(all_labels)
	print("\nF1-SCORE: ", f1_score(all_preds>0.5, all_labels))
	return mean_loss / count


def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
	best_loss = np.Inf
	best_ep = 1
	nb_iterations = len(train_loader)
	print_every = nb_iterations // 5  # print the training loss 5 times per epoch
	iters = []
	train_losses = []
	val_losses = []

	#scaler = GradScaler()
	model_counter = 0
	net.train()
	for ep in range(epochs):
		running_loss = 0.0
		for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
			# Converting to cuda tensors
			seq, attn_masks, token_type_ids, labels = seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

			opti.zero_grad()
			logits = net(seq, attn_masks, token_type_ids)

			# Computing loss
			loss = criterion(logits.squeeze(-1), labels.float())
			loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

			# Backpropagating the gradients
			loss.backward() #originially the line above is there, not this line
			opti.step()

			#if (it + 1) % iters_to_accumulate == 0: #TODO
			#lr_scheduler.step()

			running_loss += loss.item()

			if (it + 1) % print_every == 0:  # Print training loss information
				print()
				print("Iteration {}/{} of epoch {} complete. Loss : {} "
					  .format(it+1, nb_iterations, ep+1, running_loss / print_every))

				# Saving the model
				path_to_model='/home/ubuntu/mnt/cloudNAS2/SoftKBase/Google_NQ/models/{}_lr_{}_val_loss_{}_ep_{}_{}.pt'.format(bert_model, lr, round(running_loss, 5), ep, model_counter)
				torch.save(net.state_dict(), path_to_model)
				print("The model has been saved in {}".format(path_to_model))
				model_counter += 1

				val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
				print()
				print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

				running_loss = 0.0


	# Saving the model
	path_to_model='/home/ubuntu/mnt/cloudNAS2/SoftKBase/Google_NQ/models/{}_lr_{}_val_loss_{}_ep_{}_final.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
	torch.save(net.state_dict(), path_to_model)
	print("The model has been saved in {}".format(path_to_model))

	del loss
	torch.cuda.empty_cache()


# Reading train and dev data
data_dir = sys.argv[1]
train_path = open(os.path.join(data_dir,'train/train_processed.csv'))
dev_path = open(os.path.join(data_dir,'dev/dev_processed.csv'))

#train_path = open(os.path.join(data_dir,'dev/dev_processed_0.csv'))
#dev_path = open(os.path.join(data_dir,'dev/dev_processed_0.csv'))

delimiter = " <;;;> "
df_val = pd.read_csv(dev_path, delimiter=delimiter)
chunksize = 6708167
#chunksize = 100
for df_train in pd.read_csv(train_path, delimiter=delimiter, chunksize=chunksize):
	# Defining model and parameters #TODO
	bert_model = 'bert-base-uncased' 

	#bert_model = 'albert-base-v2'
	#bert_model = 'albert-large-v2' 
	#bert_model = 'albert-large-v2'
	#bert_model = 'albert-xlarge-v2'
	#bert_model = 'albert-xxlarge-v2'

	#bert_model = 'distilbert-base-uncased' 
	#bert_model = 'distilbert-base-uncased-distilled-squad'

	freeze_bert = False	# if True, freeze the encoder weights and only update the classification layer weights
	maxlen = 512  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
	bs = 4  # batch size
	iters_to_accumulate = 1  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
	lr = 1e-6  # learning rate
	epochs = 5	# number of training epochs

	#  Set all seeds to make reproducible results
	set_seed(1)

	# Creating instances of training and validation set
	print("Reading training data...")
	train_set = CustomDataset(df_train, maxlen, bert_model)
	print("Reading validation data...")
	val_set = CustomDataset(df_val, maxlen, bert_model)

	# Creating instances of training and validation dataloaders
	train_loader = DataLoader(train_set, batch_size=bs, num_workers=1)
	val_loader = DataLoader(val_set, batch_size=bs, num_workers=1)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)

	net.to(device)

	criterion = nn.BCEWithLogitsLoss()
	opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
	num_warmup_steps = 0 # The number of steps for the warmup phase.
	num_training_steps = epochs * len(train_loader)  # The total number of training steps
	t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
	lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

	train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)


