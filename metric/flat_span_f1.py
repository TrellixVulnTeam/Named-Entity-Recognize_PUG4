import json
from typing import List, Set, Tuple


def mask_span_f1(batch_preds, batch_labels, batch_masks=None, label_list: List[str] = None,
                 output_path = None):
    with open('data/saved_fig/batch_preds.txt','w') as f2:
        f2.write(str(batch_preds))
    with open('data/saved_fig/batch_labels.txt','w') as f3:
        f3.write(str(batch_labels))
   
    """
    compute  span-based F1
    Args:
        batch_preds: predication . [batch, length]预测
        batch_labels: ground truth. [batch, length]
        label_list: label_list[idx] = label_idx. one label for every position 
        batch_masks: [batch, length]

    Returns:
        span-based f1

    Examples:
        >>> label_list = ["B-W", "M-W", "E-W", "S-W", "O"]B开始，M中间词，E结束词，S单独词，O外部词
        >>> batch_golden = [[0, 1, 2, 3, 4], [0, 2, 4]]
        >>> batch_preds = [[0, 1, 2, 3, 4], [4, 4, 4]]
        >>> metric_dic = mask_span_f1(batch_preds=batch_preds, batch_labels=batch_golden, label_list=label_list)
    """
    fake_term = "一"
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    #label_list=['B-HP', 'M-HP', 'E-HP', 'S-HP', 'B-HC', 'M-HC', 'E-HC', 'S-HC', 'O']
    if batch_masks is None:
        batch_masks = [None] * len(batch_preds)#预测的长度

    outputs = []

    for preds, labels, masks in zip(batch_preds, batch_labels, batch_masks):
        if masks is not None:#但是是NONE，所以走48，49
            preds = trunc_by_mask(preds, masks)
            labels = trunc_by_mask(labels, masks)

        preds = [label_list[idx] if idx < len(label_list) else "O" for idx in preds]#如果没有标签了就用o来代替
        labels = [label_list[idx] for idx in labels]

        pred_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, pred) for pred in preds])[1]#fake_term“——”预测的标签列表
        golden_tags: List[Tag] = bmes_decode(char_label_list=[(fake_term, label) for label in labels])[1]#真实的标签列表

        pred_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in pred_tags)#set删除重复的
        golden_set: Set[Tuple] = set((tag.begin, tag.end, tag.tag) for tag in golden_tags)#tag在真实标签中
        pred_tags = sorted([list(s) for s in pred_set], key=lambda x: x[0])#因为是0所以按照第一维进行排序
        golden_tags = sorted([list(s) for s in golden_set], key=lambda x: x[0])
        outputs.append(
            {
                "preds": " ".join(preds),
                "golden": " ".join(labels),
                "pred_tags:": "|".join(" ".join(str(s) for s in tag) for tag in pred_tags),
                "gold_tags:": "|".join(" ".join(str(s) for s in tag) for tag in golden_tags)
            }
        )

        for pred in pred_set:#若在pred_set中
            if pred in golden_set:#若在真的golden_set这个tuple中，true_positive+1
                true_positives += 1
            else:                 #否则，false_positive+1
                false_positives += 1

        for pred in golden_set:#若预测在true_set中
            if pred not in pred_set:#如果不在pred_set中，false_negatives+1
                false_negatives += 1
    #print("请输出golden_tags:.............................................")
    #print(golden_tags)，变化的
    #[[90, 94, 'HC'], [100, 103, 'HC'], [105, 107, 'HC'], [107, 109, 'HC'], [112, 114, 'HC']]
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    if output_path:
        json.dump(outputs, open(output_path, "w"), indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Wrote visualization to {output_path}")

    return {
        "span-precision": precision,
        "span-recall": recall,
        "span-f1": f1
    }


def trunc_by_mask(lst: List, masks: List) -> List:
    """mask according to truncate lst"""
    out = []
    for item, mask in zip(lst, masks):
        if mask:
            out.append(item)
    return out



class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> Tuple[str, List[Tag]]:
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            idx += 1
            continue 

    sentence = "".join(term for term, _ in char_label_list)
    return sentence, tags



    

