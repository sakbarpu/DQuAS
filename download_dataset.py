'''

This script downloads and unzips the dataset.
Use it as follows:

	python download_dataset.py <data_dir>

where <data_dir> is the dir you want to store the dataset.
Make sure you have enough space. Data can be many GBs.

The data dir will look like this:

data_dir
	|___dev
	|   |__ nq-dev-00.jsonl
		...
	|   |__ nq-dev-04.jsonl
	|___train
	    |__ nq-train-00.jsonl
		...
	    |__ nq-train-49.jsonl

'''

import sys, os

data_dir = sys.argv[1]

# Where to get the data from the internel
base = "https://storage.googleapis.com/natural_questions/v1.0"
train_urls =  ["%s/train/nq-train-%02d.jsonl.gz" % (base, i) for i in range(50)] #there are 50 train files
dev_urls = ["%s/dev/nq-dev-%02d.jsonl.gz" % (base, i) for i in range(5)] #and 5 dev files

print("\nFollowing are the urls for train dataset")
print(train_urls)
print("\nFollowing are the urls for dev dataset")
print(dev_urls)

# Downloading train file from the urls
for t_url in train_urls[:1]:
	try:
		os.system("mkdir " + os.path.join(data_dir,"train"))
		os.system("wget " + t_url + " -P " + os.path.join(data_dir,"train"))
	except OSError as e:
		raise OSError("See if wget is installed")

# Downloading dev files from the urls
for d_url in dev_urls[:1]:
	try:
		os.system("mkdir " + os.path.join(data_dir, "dev"))
		os.system("wget " + d_url + " -P " + os.path.join(data_dir, "dev"))
	except OSError as e:
		raise OSError("See if wget is installed")

# Go to train dir and do unzip
try:
	os.system("cd " + os.path.join(data_dir,"train") + " ; " + "for f in nq-train-*l.gz; do echo $f; gzip -k -d $f; done;")
except OSError as e:
	raise OSError("See if gunzip is installed")

# Go to dev dir and do unzip
try:
	os.system("cd " + os.path.join(data_dir,"dev") + " ; " + "for f in nq-dev-*l.gz; do echo $f; gzip -k -d $f; done;")
except OSError as e:
	raise OSError("See if gunzip is installed")


