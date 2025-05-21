# # from quickumls.spacy_component import SpacyQuickUMLS
import spacy
from statistics import mean
import json
from scispacy.linking import EntityLinker
import spacy
from bert_score import score as b_score
from evaluate import load
# #
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
print("Loading scispacy UMLS Linker...")
# 获取并保存linker实例
linker = nlp.get_pipe("scispacy_linker")
def get_UMLS_entities(doc):

    # 初始化一个空集合来存储实体
    entities = set()

    # 遍历文档中的所有实体
    for entity in doc.ents:
        if entity._.kb_ents:
            # print(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name)
            # 提取词
            # entities.add(entity)
            # 提取词的类别
            entities.add(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name)
        else:
            entities.add("none")
    # print(entities)
    return entities

def compute_UMLS_F1(model_output, label):

    doc_model_output = nlp(model_output)
    doc_label = nlp(label)

    model_output_entities = get_UMLS_entities(doc_model_output)
    label_entities = get_UMLS_entities(doc_label)
    # Precision
    if len(model_output_entities) == 0:
        P = 0.0
    else:
        P = len([pred for pred in model_output_entities if pred in label_entities]) / len(model_output_entities)
    # Recall
    if len(label_entities) == 0:
        R = 0.0
    else:
        R = len([l for l in label_entities if l in model_output_entities]) / len(label_entities)
    # F1-score
    if (P + R) == 0:
        F = 0.0
    else:
        F = 2 * P * R / (P + R)

    return P, R, F, model_output_entities, label_entities

def compute_metrics(model_output, label, rouge):
    # capitalization is not uniform in the dataset
    predictions = [model_output.lower()]
    references = [label.lower()]

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references
    )
    # BERT_P, BERT_R, BERT_F1 = b_score(predictions, references, model_type="emilyalsentzer/Bio_ClinicalBERT")
    BERT_P, BERT_R, BERT_F1 = b_score(predictions, references, model_type="distilbert-base-uncased")
    #

    BERT_P, BERT_R, BERT_F1 = BERT_P.cpu().item(), BERT_R.cpu().item(), BERT_F1.cpu().item()

    UMLS_P, UMLS_R, UMLS_F1, model_output_entities, label_entities = compute_UMLS_F1(model_output, label)

    result_dict =  {"ROUGE_L": rouge_result["rougeL"],
            "ROUGE1": rouge_result["rouge1"],
            "ROUGE2": rouge_result["rouge2"],
            "BERT_P": BERT_P, "BERT_R": BERT_R, "BERT_F1": BERT_F1,
            "UMLS_P": UMLS_P, "UMLS_R": UMLS_R, "UMLS_F1": UMLS_F1}
    return result_dict, model_output_entities, label_entities




def open_json(path):
    with open(path, "r") as f:
        file = json.load(f)

    labels = []
    for f in file:
        labels.append(f['output'])

    return labels

def open_txt(path):
    with open(path, "r") as f:
        file = f.readlines()

    return file

def check_files(model_outputs, labels):
    print(f'lenthg for outputs: {len(model_outputs)}')
    print(f'lenthg for labels: {len(labels)}')
    assert len(model_outputs) == len(labels), 'outputs and labels must have same length'

def write_json(path, result):
	file = open(path, "w")
	json.dump(result, file)


def write_txt(path, data):
	with open(path, "a") as f:
		for y in data:
			f.write(f"{y}\n")




def compute_icd_f1(preds, gt, approximate):
    if approximate:
        preds = set([pred[:3] for pred in preds])
        gt = set([g[:3] for g in gt])

    if len(preds) == 0:
        P = 0.0
    else:
        P = len([pred for pred in preds if pred in gt]) / len(preds)
    R = len([g for g in gt if g in preds]) / len(gt)

    if (P + R) == 0:
        F = 0.0
    else:
        F = 2 * P * R / (P + R)

    return P, R, F

# def update_results(results, new_results):
#     for k in new_results:
#         if k not in results:
#             results[k] = [new_results[k]]
#         results[k].append(new_results[k])
#     return None

def compute_average_results(results):
    average_results = {}
    for k in results:
        average_results[k] = mean(results[k])
    return average_results

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

if __name__ == '__main__':
    suffix = "28"
    path_label = "../datasets/medqa_d2n/task_prefix/medqa_d2n_test2.json"
    labels = open_json(path_label)
    # print(labels)
    # suffix = '_wang2'
    path_model_output = f'./results/{suffix}/generated_predictions.txt'
    path_reulsts = f'./results/{suffix}/entities_{suffix}.json'
    log_path = f'./results/{suffix}/finalres_{suffix}.json'


    model_outputs = open_txt(path_model_output)
    # print(model_outputs)
    check_files(model_outputs, labels)

    rouge = load("rouge")

    results = {}
    # Ps = []
    # Rs = []
    # Fs = []
    model_output_entities = []
    label_entities = []
    with open(path_reulsts, "a") as out_file:
        for r in range(len(model_outputs)):
            model_output = model_outputs[r]
            label = labels[r]

            result_dict, model_output_entities, label_entities = compute_metrics(model_output, label, rouge)

            # update_results(results, result_dict)

            for k in result_dict:
                if k not in results:
                    results[k] = [result_dict[k]]
                results[k].append(result_dict[k])

            average_results = compute_average_results(results)
            print_metrics = ["ROUGE_L", "BERT_F1", "UMLS_F1"]
            result_dict["Model Answer"] = model_output_entities
            result_dict["Ground Truth Answer"] = label_entities
            print(type(result_dict))
            json.dump(result_dict, out_file, indent=4, default=set_default)
            out_file.write("\n\n")
    with open(log_path, "w") as f_w:
        json.dump(average_results, f_w, default=set_default)
