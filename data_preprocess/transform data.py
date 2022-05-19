import json 
import sys
sys.path.append("./data_preprocess")


from file_utils import load_conll 
from label_utils import get_span_labels 
from query_map import queries_for_dataset


def generate_query_ner_dataset(source_file_path, dump_file_path, entity_sign="nested",
    dataset_name=None, query_sign="default"):

    entity_queries = queries_for_dataset[dataset_name][query_sign]
    label_lst = queries_for_dataset[dataset_name]["labels"]

    if entity_sign == "nested":
        with open(source_file_path, "r") as f:
            source_data = json.load(f)
    elif entity_sign == "flat":
        source_data = load_conll(source_file_path)
    else:
        raise ValueError("ENTITY_SIGN can only be NESTED or FLAT.")

    target_data = transform_examples_to_qa_features(entity_queries, label_lst, source_data, entity_sign=entity_sign)

    with open(dump_file_path, "w") as f:
        json.dump(target_data, f, sort_keys=True, ensure_ascii=False, indent=2)


def transform_examples_to_qa_features(query_map, entity_labels, data_instances, entity_sign="nested"):

    mrc_ner_dataset = []

    if entity_sign.lower() == "flat":
        tmp_qas_id = 0 
        for idx, (word_lst, label_lst) in enumerate(data_instances):
            candidate_span_label = get_span_labels(label_lst)
            tmp_query_id = 0 
            for label_idx, tmp_label in enumerate(entity_labels):
                tmp_query_id += 1
                tmp_query = query_map[tmp_label]
                tmp_context = " ".join(word_lst)

                tmp_start_pos = []
                tmp_end_pos = []
                tmp_entity_pos = []

                start_end_label = [(start, end) for start, end, label_content in candidate_span_label if label_content == tmp_label]

                if len(start_end_label) != 0:
                    for span_item in start_end_label:
                        start_idx, end_idx = span_item 
                        tmp_start_pos.append(start_idx)
                        tmp_end_pos.append(end_idx)
                        tmp_entity_pos.append("{};{}".format(str(start_idx), str(end_idx)))
                    tmp_impossible = False 
                else:
                    tmp_impossible = True 
                
                mrc_ner_dataset.append({
                    "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                    "query": tmp_query,
                    "context": tmp_context,
                    "entity_label": tmp_label,
                    "start_position": tmp_start_pos, 
                    "end_position": tmp_end_pos,
                    "span_position": tmp_entity_pos, 
                    "impossible": tmp_impossible
                    })
            tmp_qas_id += 1 

    elif entity_sign.lower() == "nested":
        tmp_qas_id = 0 
        for idx, data_item in enumerate(data_instances):
            tmp_query_id = 0 
            for label_idx, tmp_label in enumerate(entity_labels):
                tmp_query_id += 1
                tmp_query = query_map[tmp_label]
                tmp_context = data_item["context"]

                tmp_start_pos = []
                tmp_end_pos = []
                tmp_entity_pos = []

                start_end_label = data_item["label"][tmp_label] if tmp_label in data_item["label"].keys() else -1 

                if start_end_label == -1:
                    tmp_impossible = True 
                else:
                    for start_end_item in data_item["label"][tmp_label]:
                        start_end_item = start_end_item.replace(",", ";")
                        start_idx, end_idx = [int(ix) for ix in start_end_item.split(";")]
                        tmp_start_pos.append(start_idx)
                        tmp_end_pos.append(end_idx)
                        tmp_entity_pos.append(start_end_item)
                    tmp_impossible = False 

                mrc_ner_dataset.append({
                    "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                    "query": tmp_query,
                    "context": tmp_context,
                    "entity_label": tmp_label,
                    "start_position": tmp_start_pos,
                    "end_position": tmp_end_pos,
                    "span_position": tmp_entity_pos,
                    "impossible": tmp_impossible
                    })
            tmp_qas_id += 1
    else:
        raise ValueError("Please Notice that entity_sign can only be flat OR nested. ")


    return mrc_ner_dataset



if __name__=="__main__":
    import os, sys
    generate_query_ner_dataset(entity_sign="flat",source_file_path='data/train_for_ESI/baidubaike/mrc-ner.dev', dump_file_path='data/train_for_ESI/baidubaike/mrc-ner.dev',
                               dataset_name='data/train_for_ESI/baidubaike/query_ner.dev',
                               query_sign="default")
