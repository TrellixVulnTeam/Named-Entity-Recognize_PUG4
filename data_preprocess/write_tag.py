import re
import string
import os
def write_en_tag(root_path, tag_file):#处理锚文本知识
    RE_link = re.compile(r"<a.*?>(.*?)</a>")#正则表达式
    zh_tag = open(tag_file, "w", encoding="utf-8")
    count_file = 0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            count_file += 1
            if count_file % 100 == 0:
                print(count_file)
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
                    if len(line_list) > 250:
                        continue
                    for i, j in enumerate(line_list):
                        if j[0] in string.punctuation and j[0] != "<":
                            j = j[0] + " " + j[1:]
                        if j[-1] in string.punctuation and j[-1] != ">":
                            j = j[:-1] + " " + j[-1]
                        line_list[i] = j
                    line = " ".join(line_list)


                    interlinks_raw = re.findall(RE_link, line)
                    # print(line)
                    line = re.sub(RE_link, " %%%%%%%%%%##########&&&&&******* ", line)
                    line_list = line.split()
                    a_line_list=line_list
                    interlinks_index = []
                    index_flag = 0
                    for i, j in enumerate(line_list):
                        count = j.count("%%%%%%%%%%##########&&&&&*******")
                        if count == 0:
                            continue
                        elif count == 1:
                            line_list[i] = line_list[i].replace("%%%%%%%%%%##########&&&&&*******",
                                                                interlinks_raw[index_flag])
                            index_flag += 1
                            interlinks_index.append(i)
                        else:
                            print("multiiiiiiiiiiiii")
                            for c in range(count):
                                line_list[i] = line_list[i].replace("%%%%%%%%%%##########&&&&&*******",
                                                                    interlinks_raw[index_flag],
                                                                    1)
                                index_flag += 1
                            interlinks_index.append(i)
                    tag=["O"]*len(line_list)

                    for i, j in enumerate(interlinks_index):
                        tag[j]="-Entity"

                    for i, j in enumerate(line_list):
                        if "<ahref" in j or "</a" in j:
                            continue
                        j = j.split()
                        if tag[i]=="O":
                            for k in range(len(j)):
                                zh_tag.write(j[k] + "\t" + tag[i] + "\n")
                            continue
                        if len(j) == 1:
                            zh_tag.write(j[0] + "\tS"+tag[i]+"\n")
                        else:
                            for k in range(len(j)):
                                if k==0:
                                    zh_tag.write(j[k] + "\tB"+tag[i]+"\n")
                                elif k==len(j)-1:
                                    zh_tag.write(j[k] + "\tE"+tag[i]+"\n")
                                else:
                                    zh_tag.write(j[k] + "\tI"+tag[i]+"\n")
                    zh_tag.write("\n")

                                                                                              

"""
write_zh_tag("data/zhwiki/zhwiki.simple", "data/zhwiki/zhwiki.tag")
write_zh_tag("data/enwiki/output/", "data/enwiki/enwiki.tag")
"""
#write_en_tag("/home/xinghaoran/CoFEE-main/data/train_for_ESI/baidubaike/baidubaike.html", "data_preprocess/test")


def ff(str,num):
    return str[:num] + str[num+1:]
