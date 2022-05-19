# coding=utf-8
import re
import os
import sys
import string
import json
from nltk.tokenize import sent_tokenize
import pdb

def write_en_tag(source_dir, target_dir):
    RE_link = re.compile(r'<ahref="(.*?)">(.*?)</a>')
    MASK = "%%%%%%%%%%##########&&&&&*******"
    result_dict = {}
    hash_set = set()
    count_file = 0
    out_file=open(target_dir+"enwiki.raw5-200","w")
    raw_len=0
    sent_len_dict={}
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            count_file += 1
            if count_file%10000==0:
                print("File count: " + str(count_file), end="\r")

            filename = os.path.join(root, file)
            with open(filename) as f:
                for line in f:
                    if "<doc id=" in line or "</doc>" in line:
                        continue

                    line = line.strip()
                    
                    if len(line) < 20:
                        continue
                    
                    try:
                        line = line[:-1] + " " + line[-1]
                    except:
                        continue

                    line = line.replace("<a href=", '<ahref=')
                    line_list = line.split()
                    
                    #if len(line_list) > 250:
                        #continue
                    
                    for i, j in enumerate(line_list):
                        if j[0] in string.punctuation and j[0] != "<":
                            j = j[0] + " " + j[1:]
                        if j[-1] in string.punctuation and j[-1] != ">":
                            j = j[:-1] + " " + j[-1]
                        line_list[i] = j
                    line = " ".join(line_list)

                    interlinks_raw = re.findall(RE_link, line)
                    masked_line = re.sub(RE_link, " " + MASK + " ", line)
                    sents = sent_tokenize(masked_line)
                    for sent in sents:
                        if MASK not in sent:
                            continue
                        word_list = sent.split()
                        if len(word_list)<5 or len(word_list)>200:
                            continue
                        
                        # record entity
                        interlinks_index = []
                        interlinks_entity = []
                        index_flag = 0
                        for i, word in enumerate(word_list):
                            count = word.count(MASK)
                            if count == 0:
                                continue
                            elif count == 1:
                                word_list[i] = word_list[i].replace(MASK, interlinks_raw[index_flag][1])
                                interlinks_entity.append(interlinks_raw[index_flag])
                                interlinks_index.append(i)
                                index_flag += 1
                            else:
                                print("multiiiiiiiiiiiii")
                                pdb.set_trace()
                                for c in range(count):
                                    word_list[i] = word_list[i].replace(MASK, interlinks_raw[index_flag][1], 1)
                                    interlinks_entity.append(interlinks_raw[index_flag])
                                    interlinks_index.append(i)
                                    index_flag += 1
                        
                        # create tag
                        tag = ["O"] * len(word_list)
                        for i, idx in enumerate(interlinks_index):
                            tag[idx] = interlinks_raw[i][0]

                        token_list = []
                        tag_list = []
                        entity_list = []
                        sent_len = 0
                        for i, word in enumerate(word_list):
                            if "<ahref" in word or "</a" in word:
                                continue
                            
                            tokens = word.split()

                            if tag[i] == "O":
                                for k in range(len(tokens)):
                                    token_list.append(tokens[k])
                                    tag_list.append(tag[i])
                                    sent_len += 1
                                continue

                            entity_type = tag[i]
                            if len(tokens) == 1:
                                token_list.append(tokens[0])

                            else:
                                for k in range(len(tokens)):
                                    if k == 0:
                                        token_list.append(tokens[k])

                                    elif k == len(tokens)-1:
                                        token_list.append(tokens[k])

                                    else:
                                        token_list.append(tokens[k])




                        out_file.write(" ".join(token_list)+"\n")
                        raw_len+=1
                        tmp_token_len=len(token_list)
                        if tmp_token_len in sent_len_dict:
                            sent_len_dict[tmp_token_len]+=1
                        else:
                            sent_len_dict[tmp_token_len]=1
    print(raw_len)
    print(sent_len_dict)

                            


source_dir = "/home/xuemengge/data/enwiki/output/"
target_dir = "/home/xuemengge/data/ACL2021/test_file/"
write_en_tag(source_dir, target_dir)
