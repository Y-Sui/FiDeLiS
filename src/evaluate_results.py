import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "_detailed_eval_result_top_{topk}.jsonl" if topk > 0 else '_detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('.jsonl', eval_name)
    error_file = predict_file.replace('.jsonl', '_error.jsonl')

    llm_result, direct_ans_result = {}, {}
    for item in ["prediction_llm", "prediction_direct_answer"]:
        # Load results
        acc_list = []
        hit_list = []
        f1_list = []
        precission_list = []
        recall_list = []
        error_list = []
        f = open(predict_file, 'r')
        f2 = open(detailed_eval_file, 'a')
        f3 = open(error_file, 'a')
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data['id'] if 'id' in data else "null"
            answer = data['ground_truth']
            question = data['question']
            q_entities = data['q_entities']
            reasoning_path = data['reasoning_path']
            ground_path = data['ground_path']
            prediction = data[item] if item in data else ""
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = ' '.join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            'method': item, 
                            'id': id, 
                            'prediction': prediction, 
                            'ground_truth': answer, 
                            'acc': acc, 
                            'hit': hit, 
                            'f1': f1_score, 
                            'precission': precision_score, 
                            'recall': recall_score
                            }
                        ) + '\n')
                
                if hit == 0:
                    f3.write(
                        json.dumps(
                            {
                                'method': item,
                                'id': id, 
                                'prediction': prediction, 
                                'ground_truth': answer, 
                                'question': question,
                                'q_entities': q_entities,
                                'reasoning_path': reasoning_path,
                                'ground_path': ground_path,
                                
                            }) + '\n')
                    error_list.append(id)
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'method': item, 'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit}) + '\n')
                if hit == 0:
                    f3.write(
                        json.dumps(
                            {
                                'method': item,
                                'id': id, 
                                'prediction': prediction, 
                                'ground_truth': answer, 
                                'question': question,
                                'q_entities': q_entities,
                                'reasoning_path': reasoning_path,
                                'ground_path': ground_path,
                                
                            }) + '\n')
                    error_list.append(id)
                    
        if len(f1_list) > 0:
            result_str = str(item) + " Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list)) + " Error Number: " + str(len(error_list))
        else:
            result_str = str(item) + " Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " Error Number: " + str(len(error_list))
        
        if item == "prediction_llm":
            llm_result = {
                "Accuracy": sum(acc_list) * 100 / len(acc_list),
                "Hit": sum(hit_list) * 100 / len(hit_list),
                "F1": sum(f1_list) * 100 / len(f1_list) if len(f1_list) > 0 else 0,
                "Precision": sum(precission_list) * 100 / len(precission_list) if len(precission_list) > 0 else 0,
                "Recall": sum(recall_list) * 100 / len(recall_list) if len(recall_list) > 0 else 0,
                "Error Number": len(error_list)
            }
        else:
            direct_ans_result = {
                "Accuracy": sum(acc_list) * 100 / len(acc_list),
                "Hit": sum(hit_list) * 100 / len(hit_list),
                "F1": sum(f1_list) * 100 / len(f1_list) if len(f1_list) > 0 else 0,
                "Precision": sum(precission_list) * 100 / len(precission_list) if len(precission_list) > 0 else 0,
                "Recall": sum(recall_list) * 100 / len(recall_list) if len(recall_list) > 0 else 0,
                "Error Number": len(error_list)
            }
        
        print(result_str)
        result_name = "_eval_result_top_{topk}.txt" if topk > 0 else '_eval_result.txt'
        eval_result_path = predict_file.replace('.jsonl', result_name)
        with open(eval_result_path, 'a') as f:
            f.write(result_str + '\n')
    return llm_result, direct_ans_result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str, default='results/KGQA/csqa/alpaca_default/test')
    argparser.add_argument('--cal_f1', action="store_true")
    argparser.add_argument('--top_k', type=int, default=-1)
    args = argparser.parse_args()
    
    eval_result(args.d, args.cal_f1, args.top_k)