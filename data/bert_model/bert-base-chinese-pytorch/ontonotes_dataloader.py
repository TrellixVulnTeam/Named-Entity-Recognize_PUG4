import fastNLP

data_bundle = fastNLP.io.OntoNotesNERLoader().load('/home/xuemengge/data/ACL2021/OntoNotes-5.0-NER/v4/english/')  # 返回的DataBundle中datasets根据目录下是否检测到train

#tr_data = data_bundle.get_dataset('train')
te_data = data_bundle.get_dataset('test')
print(len(te_data))
test_file=open("/home/xuemengge/data/ACL2021/OntoNotes-5.0-NER/v4/english/test.tag","w")
#dev_data=data_bundle.get_dataset('dev')
for i, d in enumerate(te_data):


    target=d["target"]
    for j,w in enumerate(d["raw_words"]):
        test_file.write(w+"\t"+target[j]+"\n")
    test_file.write("\n")

te_data = data_bundle.get_dataset('dev')
dev_file=open("/home/xuemengge/data/ACL2021/OntoNotes-5.0-NER/v4/english/dev.tag","w")
print(len(te_data))
#dev_data=data_bundle.get_dataset('dev')
for i, d in enumerate(te_data):

    target=d["target"]
    for j,w in enumerate(d["raw_words"]):
        dev_file.write(w+"\t"+target[j]+"\n")
    dev_file.write("\n")

te_data = data_bundle.get_dataset('train')
train_file=open("/home/xuemengge/data/ACL2021/OntoNotes-5.0-NER/v4/english/train.tag","w")
print(len(te_data))
#dev_data=data_bundle.get_dataset('dev')
for i, d in enumerate(te_data):

    target=d["target"]
    for j,w in enumerate(d["raw_words"]):
        train_file.write(w+"\t"+target[j]+"\n")
    train_file.write("\n")
