# DQuAS (Deep Question Answering System)

This is `dee-quas`, a deep learning system trained on [Google Natural 
Questions Dataset](https://ai.google.com/research/NaturalQuestions). 
The repository contains code to fine-tune [BERT](https://arxiv.org/abs/1810.04805)
on the training dataset and for prediction on the dev dataset.

## Google Natural Questions Dataset

The Google NQ dataset is a question answering dataset that
can be thought of as a triple (question, long answer, short
answer).

The questions are extracted from Google query logs and
contain keywords queries that were posted on the Google 
search engine. In response to the query, a list of websites
are returned by the Google search engine. The top wikipedia
articles that are retrieved in response to a given query are
given to the human annotators whose job is to label passages
in the article that form the long answers to the Google
query. In addition to the long answer, the annotator is also
responsible to label a short answer (most like a couple of
words or a small phrase) in the Wikipedia article.

The goal then is given a Google query and a Wikipedia
article with many passages (may be 100 passages), find the 
long answer and short answer to the query.

In this DQuAS project, we are interested in only localizing
the passage that constitute the long answer to the query.
And we fine-tune BERT to perform binary sentence pair 
classification task. In the sentence pair, one item is the
question text, while the other is the long answer candidate.
The labels will be 0 for candidates that are not actual
long answer, while 1 for actual long answer passage.


## Instructions for installing and running code 
### Prerequisites
  * Ubuntu 16.04 instance with CUDA installed
  * Python 3.7
  * Pandas 0.24.0
  * Sklearn 0.20.1
  * **Torch 1.7.0**
  * **Huggingface Transformers 2.4.0**
  * wget and gzip Linux tools for downloading and unzipping data files
  * **Hardware: Intel Core Processor with NVIDIA GTX 1080 Ti**
  
### Downloading data
  Simply use the download_dataset.py script to download Here is how to use the script:
    
    python download_dataset.py <data_dir>
  
  where `<data_dir>` is the place where you want to hold the data. Make sure there is
  enough space on disk because dataset can be many GBs.
  
  It will store the data as follows:
   
  ```
  data_dir
        ├── dev
        │   ├── nq-dev-00.jsonl
                ...
        │   └── nq-dev-04.jsonl
        └── train
            ├── nq-train-00.jsonl
                ...
            ├── nq-train-49.jsonl
  ```

### Preprocessing data
  With a single python command you can preprocess the data (both train and dev)
  
    python preprocess_data.py <data_dir>
    
  where `<data_dir>` is the place where you want to hold the data. The result will be 
  creation of two csv files train_processed.csv and dev_processed.csv.
  
  ```
  data_dir
        |_ train
                |_ train_processed.csv
        |_ dev
                |_ dev_processed.csv
  ```

The csv files will contain the following columns: question, label, 
and candidate. Here, question is the text sequence for question, candidate is the
long answer text candidate, and label is 1 for train example where candidate is 
the actual long answer, and 0 otherwise.



### Train the model
 Again with a single command you can train the model.
 
   ```
   python train.py <data_dir> <model_dir>
   ```
   
 where `<data_dir>` is the path to the root of the data. And `model_dir` refers to the
 place you want to store the model checkpoints.
 
### Prediction
 This is where we run the trained model to perform predictions on the dev set and produce
 the predictions file as required by the nq_eval.py file.
 
   ```
   python prediction.py <data_dir> <model_path> <output_file>
   ```
   
 where `<data_dir>` is the root dir of the data, `<model_path>` is the path to the checkpoint
 of the model that you want to use for prediction, and `<output_file>` is the output json file
 path where you want to store the prediction results.
 
 In `<data_dir>` this script will also create a file called dev_processed_new.csv. 
 The dev_processed.csv file will contain the following columns: question, label,
candidate, start, and end. Here, question, label, and candidate has the same definition
as above, whereas start and end are the start byte and end byte of the candidate
in the Wikipedia article. The start byte is the byte index from where the candidate
span starts in the Wikipedia article, and end byte is the byte index at which the 
candidate span ends. These two bytes are needed for evaluation of the dev set through
the Google provided nq_eval.py script (more details later).
 
 ### Metric calculation using nq_eval.py 
 We use the [nq_eval.py](https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py)
 script provided by the authors of the dataset to evaluate the performance of our model on
 the dev predictions.
 
 The way to call this script is straightforward:
 
 ```
   python nq_eval.py --gold_path=<path-to-gold-files> --predictions_path=<path_to_json>
 ```
 
 where `<gold_path>` is the list of raw gunzipped dev data files that were downloaded from 
 the internet and `<predictions_path>` is the output json file generated by `prediction.py` script.
  
## F1 score / Precision / Recall when running nq\_eval.py for the task

When evaluation script nq_eval.py is run against the predictions produced by the model in
prediction.py and original gold dev files, it produces the following metric values for long
answer prediction:

| Metric | Value|
---------|----------------------------------------------------
"long-best-threshold-f1" | 0.28950159066808057
"long-best-threshold-precision" | 0.2441860465116279
"long-best-threshold-recall" | 0.35546875                         
"long-best-threshold" | 0.39462462067604065                 
"long-recall-at-precision>=0.5" | 0.025824652777777776                   
"long-precision-at-precision>=0.5" | 0.5085470085470085 
"long-recall-at-precision>=0.75" | 0.0006510416666666666                 
"long-precision-at-precision>=0.75" | 1.0   
"long-recall-at-precision>=0.9" | 0.0006510416666666666                  
"long-precision-at-precision>=0.9" | 1.0                   

## Replicate these results

 Use the setting currently implemented there in the code and the results should be replicable.
 Also, we could only go up until the second checkpoint and so just used that one for prediction.
 After second checkpoint (and almost 2 days of training) we killed the training process.
 Otherwise, it would have taken more than a week to train on GTX 1080 Ti.
 
 We divide the data (53665339 training examples) into 8 chunks and train on each chunk (size = 
 6708167 examples) mainly because we can't load all examples simultaneously into my memory.
 
Then, we take checkpoints of model at every 20% of the training of each chunk (i.e, around 
1341633 examples). We trained for just two checkpoints (i.e. 40%) of the first chunk. That makes 
the effective training for 1341633\*2 =  2683266 examples out of total 53665339 examples.
 
In all, the training is done for just (2683266 out of 53665339 examples or) 5% of our processed
binary classification data. It took approximately 54 hours to train the system.
 
 The following are the training criteria (hyperparamters etc.) used:
 
    
    freeze_bert = False     # if True, freeze the encoder weights and only update the classification layer weights
    maxlen = 512  # max length of the tokenized input sentence pair : if > "maxlen", the input is truncated, else if smaller, the input is padded
    bs = 4  # batch size
    lr = 1e-6  # learning rate
    criterion = nn.BCEWithLogitsLoss() # this is binary cross entropy with logits loss
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)   
    

## The Neural Network Architecture 

We have used the Huggingface library to load the pretrained BERT model for fine-tuning on our
task. In the code, there is capability of playing with multiple different pretrained model.
In current state the model used is the [`bert-based-uncased`](https://huggingface.co/bert-base-uncased)
model.

The BERT model has 110M parameters and is pretrained on a large text corpus of Wikipedia and news 
articles on two different tasks (1) MLM (masked language model) and (2) Next sentence prediction.

For fine-tuning on our binary classification task, we provide (question, candidate) pairs as 
input sequence and use the `[CLS]` embedding for classification. The embedding for this token
is provided as input to a Linear layer with just one sigmoid output. The sigmoid output tells us
probability that there is a match between question and candiate. We can threshold the probability 
estimate to determine whether the prediction is positive (1) or negative (0). For the training examples, 
the model is trained using (question, candidate) pairs.

The binary classifier is trained using binary cross entropy loss or `BCEWithLogitsLoss` in PyTorch with
`AdamW` optimizer.

### Why it works?

This architecture is simple and naturally works for this binary classification problem. The problem
is naturally a binary classification problem of finding relevance/matching score between two pieces 
of texts question and candidate needs to be determined. BERT binary pair of sentence classification 
works best for this type of problem. Since, BERT is pretrained on next-sentence prediction task, it
makes it suitable for this kind of problem. 

### Sampling Heuristic

However, for each positive example (question, long answer) pair, there can be many negative examples
(question, candidate) pairs. If there are hundreds of such negative candidates then training becomes
prohibitively compute expensive. Therefore, we rely on negative sampling heuristic defined below.

We sample 10% of negative example pairs for each positive examples. So, if there are 100 negative
examples, we sample 10 of them. But, this is not just a random sampling with uniform distribution.
Rather, it's sampling based on the length of the candidate in the negative example pair.

Specifically, we sample 50% or 5 examples that have the candidate closest in length to the actual 
positive example, 25% or 3 examples that have the candidate smallest in length, and the rest 25%
or 3 examples that have the candidate largest in length. This type of sampling is performed to
ensure that we sample different lengths of candidate and also to ensure that most of the samples
have their lengths closest to the actual positive long answer.

## Improvements possible

  * We can try BERT-ensembles. They are performing well in SOTA.

  * We can try more recent deep learning models for training instead of BERT. 
  
  * We can also try fine-tuning BERT for longer periods of time. We have only trained it for a VERY SMALL portion of dataset. 

  * We can use bigger BERT model (bert-large).
  
  * Knowledge-aware models like RAG could be used to improve performance further. These models use knowledge-base
    as guiding force for making decisions about relevance of a text piece to a query.

  * Multi-GPU training is necessary for this kind of task for fast processing.
  
  * We can explore the dataset for types of questions asked by the user to determine the user intent.
    Based on that, we can first identify what kind of question the user is asking and then build a separate
    model for each type of intent. Also, if the intent is not decidable, we can use straightforward modeling
    approach.
    
  * Efficiency needs to be improved. For this task where only upto several hundred candidate passages are there
    in a Wikipedia article, it's tolerable to have binary classification with the input pairs (question, candidate)
    passing through the whole neural network to get matching score. In reality, it will be more efficient to change
    the architecture in a way that at the inference time all you have to do is compare create query embedding and 
    compare it with the document embeddings in the corpus. These doc embeddings would serve as "indexes" for docs.

## Critical analysis of the Google NQ task

In a practical scenario, there is definitely one more task that has to happen 
before this Google NQ task. And that is the retrieval of the relevant Wikipedia article.
This Google NQ task assumes we already know the article we want to find the passage in
which is not very realistic since there could be hundreds of articles that would have relevance
to the question.

Also, it shouldn't be just about Wikipedia articles, rather a multi-domain (News, Blogs, Code even)
dataset should be more complex/challenging and should be a true reflection of human-level performance
on search and retrieval tasks.

## Please donate to:

[UNICEF](https://www.unicefusa.org/mission/emergencies/child-refugees-and-migrants?form=FUNSUJMLZDZ&utm_content=taxdeduct1responsive_E2001&ms=cpc_dig_2020_Brand_20200109_google_taxdeduct1responsive_delve_E2001&initialms=cpc_dig_2020_Brand_20200109_google_taxdeduct1responsive_delve_E2001&gclid=Cj0KCQiAzsz-BRCCARIsANotFgN5fgFXSgUWaUHVRpfO37gI2DULk_Aqco9x2JrK4LNYUNhCz_cGebMaApc3EALw_wcB)
