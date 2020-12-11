# DQuAS (Deep Question Answering System)

## Instructions for installing and running code 
### Prerequisites
  * Need to have a Ubuntu instance with CUDA installed.
  * Python 3.7
  * Pandas 0.24.0
  * Sklearn
  * Torch 1.7.0
  * Huggingface Transformers 2.4.0
  * wget and gzip for downloading and unzipping train and dev files
  
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

## The Neural Network Architecture 

### Why this architecture

## Improvements possible (full time for a month)

## Improvements in terms of hardware

## Opinion on the task and any suggestions you may have for improving it

## Favorite charity and a link to the donation page

