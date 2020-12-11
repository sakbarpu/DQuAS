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
  creation of two csv files trained_processed.csv and dev_processed.csv.
  
  ```
  data_dir
        |_ train
                |_ train_processed.csv
        |_ dev
                |_ dev_processed.csv
  ```

## F1 score / Precision / Recall when running the evaluation script nq\_eval.py for the task

## Replicate these results

## The Neural Network Architecture 

### Why this architecture

## Improvements possible (full time for a month)

## Improvements in terms of hardware

## Opinion on the task and any suggestions you may have for improving it

## Favorite charity and a link to the donation page

