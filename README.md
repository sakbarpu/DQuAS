# DQuAS (Deep Question Answering System)

## Instructions for installing and running code 
### Prerequisites
  * Need to have a Ubuntu instance with CUDA installed.
  * Python 3.7
  * Pandas 0.24.0
  * Sklearn
  * **Torch 1.7.0**
  * **Huggingface Transformers 2.4.0**
  * wget and gzip for downloading and unzipping train and dev files
  * **Hardware: Intel Core Processor with NVIDIA GTX 1080 Ti**
  
### Downloading data
  Simply use the download_dataset.py script to download Here is how to use the script:
    
    python download_dataset.py <data_dir>
  
  where `<data_dir>` is the place where you want to hold the data. Make sure there is
  enough space on disk because dataset can be many GBs.

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
 
## F1 score / Precision / Recall when running the evaluation script nq\_eval.py for the task

When evaluation script nq_eval.py is run against the predictions produced by the model in
prediction.py and original gold dev files, it produces the following metric values for long
answer prediction:

| Metric | Value|
---------|----------------------------------------------------
"long-best-threshold-f1" | 0.9071358748778103               
"long-best-threshold-precision" | 0.8300536672629696        
"long-best-threshold-recall" | 1.0                         
"long-best-threshold" | 0.9999969005584717                 
"long-recall-at-precision>=0.5" | 1.0                      
"long-precision-at-precision>=0.5" | 0.8300536672629696    
"long-recall-at-precision>=0.75" | 1.0                     
"long-precision-at-precision>=0.75" | 0.8300536672629696   
"long-recall-at-precision>=0.9" | 0                        
"long-precision-at-precision>=0.9" | 0                     

## Replicate these results

 Use the setting currently implemented there in the code and the results should be replicable.
 Also, I could only go up till the second checkpoint and so just used that one for prediction.
 Otherwise, it would have taken more than a week to train on GTX 1080 Ti. Follow the first 
 section for instructions on how to run the program.

## The Neural Network Architecture 

We have used the Huggingface library to load the pretrained BERT model for fine-tuning on our
task. In the code, there is capability of playing with multiple different pretrained model.
In current state the model used is the `bert-based-uncased`model.

For fine-tuning on our binary classification task, we provide (question, candidate) pairs as 
input sequence and use the `[CLS]` Layer used for classification. The embedding for this layer
is sent as input to a Linear layer with just one sigmoid output. The sigmoid output tells us
whether the prediction is positive (1) or negative (0). For the training examples, the model
is trained using (question, candidate) pairs.

This architecture is simple and naturally works for this binary classification problem.

## Improvements possible (full time for a month)

Yes, we can try more recent deep learning models for training instead of BERT. We can also try
fine-tuning BERT for longer periods of time. I have only trained it for a portion of dataset.

Also, knowledeg-aware models like RAG could be used to improve performance further. 

## Improvements in terms of hardware

I think multi-GPU training is necessary for this kind of task for fast processing.

## Opinion on the task and any suggestions you may have for improving it

There is definitely one more layer on top of this task that has to happen 
before this task. And that is the retrieval of the relevant Wikipedia article.
This task assumes we already know the article we want to find passage in.

Also, it shouldn't be just about Wikipedia, rather multi-domain (News, Blogs, Code even).

## Please donate to:

[UNICEF](https://www.unicefusa.org/mission/emergencies/child-refugees-and-migrants?form=FUNSUJMLZDZ&utm_content=taxdeduct1responsive_E2001&ms=cpc_dig_2020_Brand_20200109_google_taxdeduct1responsive_delve_E2001&initialms=cpc_dig_2020_Brand_20200109_google_taxdeduct1responsive_delve_E2001&gclid=Cj0KCQiAzsz-BRCCARIsANotFgN5fgFXSgUWaUHVRpfO37gI2DULk_Aqco9x2JrK4LNYUNhCz_cGebMaApc3EALw_wcB)
