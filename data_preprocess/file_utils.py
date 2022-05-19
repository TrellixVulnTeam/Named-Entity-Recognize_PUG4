def export_conll(sentence, label, export_file_path, dim=2):
    """
    参数:
        序列: 列表 [["北", "京", "天", "安", "门"], ["真", "相", "警", 告"]]
        标签:列表[["B", "M", "E", "S", "O"], ["O", "O", "S", "S"]] 开始位置打B中间位置打M结束位置打E嵌套实体打S其他位置打上o
    """
    with open(export_file_path, "w") as f:
        for idx, (sent_item, label_item) in enumerate(zip(sentence, label)): 
            for char_idx, (tmp_char, tmp_label) in enumerate(zip(sent_item, label_item)):
                f.write("{} {}\n".format(tmp_char, tmp_label))
            f.write("\n")


def load_conll(data_path):
    """
        [([word1, word2, word3, word4], [label1, label2, label3, label4]), 
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag 
        for line in f:
            if line != "\n":
                # line = line.strip()
                line=line.split()
                if len(line)==1:
                    word=","
                    tag=line[0]
                else:
                    word=line[0]
                    tag=line[1]
                #word, tag = line.split()
                word = word.strip()
                tag = tag.strip()
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raise! skipping a word")
            else:

                if len(words) > 0 and len(words)<100:
                    #print(len(words))
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                words, tags = [], []

    return dataset 


def dump_tsv(data_lines, data_path):
    """
    Desc:
        数据以tsv格式存储成TAGGING
    输入:
        data_lines的格式是:
            [([word1, word2, word3, word4], [label1, label2, label3, label4]), 
            ([word5, word6, word7, word8, word9], [label5, label6, label7, label8, label9]), 
            ([word10, word11, word12, ], [label10, label11, label12])]
    """
    print("dump dataliens into TSV format : ")
    with open(data_path, "w") as f:
        for data_item in data_lines:
            data_word, data_tag = data_item 
            data_str = " ".join(data_word)
            data_tag = " ".join(data_tag)
            f.write(data_str + "\t" + data_tag + "\n")
        print("dump data set into data path")
        print(data_path)




