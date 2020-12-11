import os

base = "https://storage.googleapis.com/natural_questions/v1.0"
train_urls =  ["%s/train/nq-train-%02d.jsonl.gz" % (base, i) for i in range(50)]
dev_urls = ["%s/dev/nq-dev-%02d.jsonl.gz" % (base, i) for i in range(5)]

print(train_urls)
print(dev_urls)

#for t_url in train_urls:
#	os.system("wget " + t_url + " -P " + "train")

for d_url in dev_urls:
	os.system("wget " + d_url + " -P " + "dev")

#make sure gzip is installed

#go to train dir and do
#for f in nq-train-*l.gz; do echo $f; gzip -d $f; done;

#go to dev dir and do
#for f in nq-dev-*l.gz; do echo $f; gzip -d $f; done;	

