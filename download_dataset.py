import sys, os

data_dir = sys.argv[1]

# Where to get the data from the internel
base = "https://storage.googleapis.com/natural_questions/v1.0"
train_urls =  ["%s/train/nq-train-%02d.jsonl.gz" % (base, i) for i in range(50)] #there are 50 train files
dev_urls = ["%s/dev/nq-dev-%02d.jsonl.gz" % (base, i) for i in range(5)] #and 5 dev files

print("Following are the urls for train dataset")
print(train_urls)
print("Following are the urls for dev dataset")
print(dev_urls)

# Downloading train file from the urls
for t_url in train_urls:
	try:
		os.system("wget " + t_url + " -P " + os.path.join(data_dir,"train"))
	except OSError as e:
		raise OSError("See if wget is installed")

# Downloading dev files from the urls
for d_url in dev_urls
	try:
		os.system("wget " + d_url + " -P " + os.path.join(data_dir, "dev"))
	except OSError as e:
		raise OSError("See if wget is installed")

# Go to train dir and do unzip
try:
	os.system("cd " + os.path.join(data_dir,"train") + " ; " + "for f in nq-train-*l.gz; do echo $f; gzip -d $f; done;")
except OSError as e:
	raise OSError("See if gunzip is installed")

# Go to dev dir and do unzip
try:
	os.system("cd " + os.path.join(data_dir,"dev") + " ; " + "for f in nq-dev-*l.gz; do echo $f; gzip -d $f; done;")
except OSError as e:
	raise OSError("See if gunzip is installed")
