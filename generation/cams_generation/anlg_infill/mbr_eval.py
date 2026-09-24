import os, sys, json
import glob
from functools import partial
sys.path.insert(0, 'e2e-metrics')
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from metrics.pymteval import BLEUScore, NISTScore
from nltk.translate.meteor_score import meteor_score
from parse import *
import json
import sys, os, torch
from spacy.lang.en import English
import ast
from transformers import BertForMaskedLM, BertTokenizer

MODE = sys.argv[1] # ar or diff
SPLIT = sys.argv[2] # val or test
OUT_PATH = sys.argv[3] # output path.
INPUT_PATH = sys.argv[4] 
def load_results_simple(path):
    with open(path, 'r') as f:
        full_result_dict = json.load(f)
    return full_result_dict



def load_results(sent_lst, tokenizer):
    # target_file = f"{INPUT_PATH}_*.json"
    # target_file = glob.glob(target_file)
    # print([x for x in target_file if 'val' not in x and 'test' not in x])
    # 10/0
    full_result_dict = {}
    failed_instances = []
    found_idx = []
    sent_lst_lst = list(sent_lst.items())
    for idx, (key, val) in enumerate(sent_lst_lst):
        # if idx < 2500: continue
        if idx in full_result_dict.keys(): continue
        word_lst1 = [x.text for x in tokenizer(val['obs1'])]
        word_lst2 = [x.text for x in tokenizer(val['obs2'])]
        target_file = f"{INPUT_PATH}_*_{SPLIT}_{idx}.json"

        file_lst = glob.glob(target_file)
        # print(file_lst, target_file)
        try:
            assert len(file_lst) == 1
        except:
            print('the file must have existed in a batched version')
            # if SPLIT == 'val': assert False
            # if idx % 100 == 1: idx = idx-1
            target_file = f"{INPUT_PATH}_*_{idx}.json"
            file_lst = glob.glob(target_file)
            print(file_lst, target_file)
            print(file_lst)
        target_file = file_lst[0]
        if "x128" in target_file:
            infill_lst = []
            with open(target_file, 'r') as f:
                for line in f:
                    example = json.loads(line)[0]
                    infill_ = example.split()[len(word_lst1):-len(word_lst2)]
                    # print(len(infill_))
                    # print(infill_, example)
                    # assert len(infill_) == 10
                    infill_=' '.join(infill_)
                    # print(infill_)
                    infill_lst.append(infill_)
            result_dict = {
                "pred_samples": infill_lst,
                "sample": None,
                "obs1": val['obs1'],
                "obs2": val['obs2']
            }
            full_result_dict[idx] = result_dict
        else:
            with open(target_file, 'r') as f:
                for line in f:
                    example = ast.literal_eval(line.strip())
                    index, template = list(example.keys())[0]
                    print(index, idx)
                    if int(index) < int(idx):
                        continue
                    assert int(index) == int(idx)
                    found_idx.append(idx)
                    example = list(example.values())[0]
                    kk, val = sent_lst_lst[idx]
                    word_lst1 = [x.text for x in tokenizer(val['obs1'])]
                    word_lst2 = [x.text for x in tokenizer(val['obs2'])]
                    infill_lst = [" ".join(xx.split()[len(word_lst1):-len(word_lst2)]) for xx in example]
                    result_dict = {
                        "pred_samples": infill_lst,
                        "sample": None,
                        "obs1": val['obs1'],
                        "obs2": val['obs2']
                    }
                    full_result_dict[idx] = result_dict
                    idx += 1

    with open('full_diff_test_outputs_aug.json', 'w') as f:
        json.dump(full_result_dict, f)
    return full_result_dict


# read files.
def mbr(result_lst, total_len, sample_size, utility):
    result = []
    for i in range(total_len):
        example_set = result_lst[i * sample_size:(i + 1) * sample_size]
        # print(example_set)
        score_dict = {}
        for idx in range(len(example_set)):
            y = example_set[idx]
            utility_lst = []
            for idx_x in range(len(example_set)):
                if idx_x != idx:
                    utility_lst.append(utility(example_set[idx_x], y))
            score_dict[idx] = np.array(utility_lst).mean()
        # print(score_dict)
        best_y = sorted(score_dict.items(), key=lambda item: item[1])[-1]
        result.append(example_set[best_y[0]])
        # print(best_y)

    return result


def bleu_score(scorer, sent_sys, sents_ref):
    scorer.reset()
    scorer.append(sent_sys, [sents_ref])
    return scorer.score()


def meteor_score2(pred, ref):
    meteor = meteor_score([ref.split()], pred.split())
    return meteor

def apply_mbr_func(full_result_dict, outpath, sent_lst):
    assert len(sent_lst) == len(full_result_dict)
    out_handle = open(outpath, 'w')
    count = 0
    for idx, val in full_result_dict.items():
        infill_lst = val['pred_samples']
        print(count, idx )
        assert count == int(idx)
        count += 1
        sample_size = len(infill_lst)
        total_len = 1
        mteval_scorers = [BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()]
        result_lst = mbr(infill_lst, total_len, sample_size, partial(bleu_score, mteval_scorers[1]))
        print(infill_lst)
        print(result_lst)
        result_str = result_lst[0]
        result_dict = {
            "pred_samples": infill_lst,
            "sample": result_str,
            "obs1": val['obs1'],
            "obs2": val['obs2']
        }
        print(json.dumps(result_dict), file=out_handle)
    out_handle.close()
    print(f'written to {outpath}')
    return

if SPLIT == 'val':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/dev_cleanup.json'
elif SPLIT == 'test':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/test_cleanup_no_label.json'
else:
    assert False, "invalid split"

with open(source_file, 'r') as f:
    sent_lst = json.load(f)



if MODE == 'diff':
    nlp = English()
    tokenizer = nlp.tokenizer
    # load_results(sent_lst, tokenizer)
    # 10/0
    decoded_dict = load_results_simple(INPUT_PATH)
    ############3
    # small_decoded_dict = {}
    # for i in range(10):
    #     small_decoded_dict[i] = decoded_dict[str(i)]
    # decoded_dict = small_decoded_dict
    # small_sent_lst = {}
    # for k, v in sent_lst.items():
    #     if len(small_sent_lst) > 9: break
    #     small_sent_lst[k] = v
    # sent_lst = small_sent_lst
    ############3
    outpath = OUT_PATH
    apply_mbr_func(decoded_dict, outpath, sent_lst)


elif MODE == 'ar':
    outpath = OUT_PATH 
    out_handle = open(outpath, 'w')
    sample_file = INPUT_PATH 
    nlp = English()
    tokenizer = nlp.tokenizer
    print(len(sent_lst))
    sample_lst = []
    with open(sample_file, 'r') as f:
        for line in f:
            sample_dict = json.loads(line)
            sample_lst.append(sample_dict)

    for idx, (key, val) in enumerate(sent_lst.items()):
        # if idx < 109: continue
        # if idx > 499:
        #     break
        infill_lst = sample_lst[idx]['samples']
        sample_size = len(infill_lst)
        total_len = 1
        mteval_scorers = [BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()]
        result_lst = mbr(infill_lst, total_len, sample_size, partial(bleu_score, mteval_scorers[1]))
        print(infill_lst)
        print(result_lst)
        result_str = result_lst[0]
        result_dict = {
            "pred_samples": infill_lst,
            "sample": result_str,
            "obs1": val['obs1'],
            "obs2": val['obs2']
        }
        print(json.dumps(result_dict), file=out_handle)

    out_handle.close()
    print(f'written to {outpath}')


# print(file+'.clean')
# with open(file+'.clean', 'w') as f:
#     for line in result_lst:
#         print(line, file=f)

