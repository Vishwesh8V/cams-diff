import torch, json, sys 

SPLIT = sys.argv[1] # val or test
MBR_PATH = sys.argv[2] # output path.

# read files.
if SPLIT == 'val':
    source_file = '/diffusion_lm/ROCstory/anlg/anlg/dev_cleanup.json'
elif SPLIT == 'test':
    source_file = '/diffusion_lm/ROCstory/anlg/anlg/test_cleanup_no_label.json'
else:
    assert False, "invalid split"

with open(source_file, 'r') as f:
    sent_lst = json.load(f)

# read generation
generated_lst = []

with open(MBR_PATH, 'r') as f:
    for line in f:
        generated_lst.append(json.loads(line))

print(len(generated_lst), len(sent_lst))

if SPLIT == 'val':
    eval_file_gen = f"{MBR_PATH}_gen.txt"
    fgen = open(eval_file_gen, 'w')
    eval_file_gold = f"{MBR_PATH}_ref.txt"  
    fgold = open(eval_file_gold, 'w')
    for gen, gold in zip(generated_lst, sent_lst.items()):
        print(gen['sample'], file=fgen)
        gold = gold[1]
        for x in gold['gold_labels']:
            print(x, file=fgold)
        print('', file=fgold)
    fgold.close()
    fgen.close()
elif SPLIT == 'test':
    eval_file_prediction = f"{MBR_PATH}_prediction.json"  
    # fpred = open(eval_file_prediction, 'w')
    full_dict = {}
    for gen, gold in zip(generated_lst, sent_lst.items()):
        print(gold)
        print(gen['sample'])
        full_dict[gold[0]] = gen['sample']
        # temp_dict = {gold[0]:gen['sample']}
        # print(temp_dict)
        # print(json.dumps(temp_dict), file=fpred)
        # gold = gold[1]
        # for x in gold['gold_labels']:
        #     print(x, file=fgold)
        # print('', file=fgold)
    with open(eval_file_prediction, 'w') as fpred:
        json.dump(full_dict, fpred)

    ###########
    test_ref = '/diffusion_lm/ROCstory/anlg/anlg/test_cleanup_ref.json'
    with open(test_ref, 'r') as f:
        test_ref_lst = json.load(f)

    eval_file_gen = f"{MBR_PATH}_gen.txt"
    fgen = open(eval_file_gen, 'w')
    eval_file_gold = f"{MBR_PATH}_ref.txt"
    fgold = open(eval_file_gold, 'w')
    for gen, gold in zip(generated_lst, sent_lst.items()):
        story_id = gold[0]
        print(gen['sample'], file=fgen)
        for x in test_ref_lst[story_id]:
            print(x, file=fgold)
        print('', file=fgold)
    fgold.close()
    fgen.close()


# generate prediction.json

