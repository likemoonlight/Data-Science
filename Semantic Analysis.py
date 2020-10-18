# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 03:23:39 2019

@author: jason
"""
from __future__ import absolute_import, division, print_function
import torch
import csv
import os
import sys
import logging
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
import pandas as pd
from tqdm import tqdm_notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from multiprocessing import Pool, cpu_count
##########################################################################################################################################
#class 
class InputF(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
class InputBERT(object):    
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class DataPro(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lin = []
            for li in reader:
                if sys.version_info[0] == 2:
                    li = list(unicode(cell, 'utf-8') for cell in li)
                lin.append(li)
            return lin
class BCP(DataPro):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")    
    def get_test_examples(self, data_dir):
        return self._create_test_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    def get_labels(self):
        return ["0", "1"]
    def _create_examples(self, lin, set_type):
        examples = []
        for (i, li) in enumerate(lin):
            guid = "%s-%s" % (set_type, i)
            text_a = li[3]
            label = li[1]
            examples.append(
                InputBERT(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    def _create_test_examples(self, lin, set_type):
        examples = []
        for (i, li) in enumerate(lin):
            guid = "%s-%s" % (set_type, i)
            text_a = li[2]
            label = 2
            examples.append(
                InputBERT(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples  
##########################################################################################################################################
def convert_feature(example_row):
    example, label_map, max_seq_length, tokenizer, output_mode = example_row
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        truncate_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)    
    return InputF(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,label_id=label_id,)  
def truncate_pair(tokens_a, tokens_b, max_length):    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train 
df_training = pd.read_csv("train.csv").drop(["date"],axis =1)
df_training = df_training.sample(frac=0.6, random_state=2019).reset_index(drop=True)
train_df_bert = pd.DataFrame({
    'index':range(len(df_training)),
    'label':df_training["label"],
    'alpha':['a']*df_training.shape[0],
    'text': df_training["text"].replace(r'\n', ' ', regex=True)})
train_df_bert = train_df_bert[['index','label','alpha',"text"]]
train_df_bert.head()
#validation
df_v = df_training.sample(frac=0.4, random_state=2019).reset_index(drop=True)
dv_bert = pd.DataFrame({
    'id':range(len(df_v)),
    'label':df_v["label"],
    'alpha':['a']*df_v.shape[0],
    'text': df_v["text"].replace(r'\n', ' ', regex=True)})
dv_bert = dv_bert[['id','label','alpha',"text"]]
dv_bert.head()
##########################################################################################################################################
#test
df_t = pd.read_csv("test.csv").drop(["date"],axis =1)
dt_bert = pd.DataFrame({
    'id':range(len(df_t)),
    'alpha':['a']*df_t.shape[0],
    'text': df_t["text"].replace(r'\n', ' ', regex=True)})
dt_bert = dt_bert[['id','alpha',"text"]]
dt_bert.head()
train_df_bert.to_csv('data/train.tsv', sep='\t', index=False, header=False)
dv_bert.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
dt_bert.to_csv('data/test.tsv', sep='\t', index=False, header=False)
logger = logging.getLogger()
csv.field_size_limit(2147483647) 
#input data
DATA_DIR = "data/"
BERT_MODEL = 'bert-base-cased'
TASK_NAME = 'sentimatic_analysis'
OUTPUT_DIR = f'outputs/{TASK_NAME}/'
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'
CACHE_DIR = 'cache/'
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'regression'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
output_mode = OUTPUT_MODE
cache_dir = CACHE_DIR
if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
processor = BCP()
train_examples = processor.get_train_examples(DATA_DIR)
train_examples_len = len(train_examples)
label_list = processor.get_labels() 
num_labels = len(label_list)
num_train_optimization_steps = int(
    train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)   
##########################################################################################################################################
#train
label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in train_examples]
process_count = cpu_count() - 1
if __name__ ==  '__main__':
    print(f'Preparing to convert {train_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        train_features = list(tqdm_notebook(p.imap(convert_feature, train_examples_for_processing), total=train_examples_len))
with open(DATA_DIR + "train_features.pkl", "wb") as f:
    pickle.dump(train_features, f)
#val
val_examples = processor.get_dev_examples(DATA_DIR)
val_examples_len = len(val_examples)
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)
label_map = {label: i for i, label in enumerate(label_list)}
val_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in val_examples]
process_count = cpu_count() - 1
if __name__ ==  '__main__':
    print(f'Preparing to convert {val_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        val_features = list(tqdm_notebook(p.imap(convert_feature,val_examples_for_processing), total=val_examples_len))
with open(DATA_DIR + "val_features.pkl", "wb") as f:
    pickle.dump(val_features, f)       
#test
processor = BCP()
test_examples = processor.get_test_examples(DATA_DIR)
test_examples_len = len(test_examples)
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)
label_map = {label: i for i, label in enumerate(label_list)}
test_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in test_examples]
process_count = cpu_count() - 1
if __name__ ==  '__main__':
    print(f'Preparing to convert {test_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        test_features = list(tqdm_notebook(p.imap(convert_feature,test_examples_for_processing), total=test_examples_len))
with open(DATA_DIR + "test_features.pkl", "wb") as f:
    pickle.dump(test_features, f)    
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=1)
model.to(device)   
# bert 
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)
##########################################################################################################################################
global_step = 0
nb_tr_steps = 0
tr_loss = 0
logger.info("***** Running training *****")
logger.info("  Num examples = %d", train_examples_len)
logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
logger.info("  Num steps = %d", num_train_optimization_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)
#BERT model
model.train()
print(OUTPUT_MODE)
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits = model(input_ids, segment_ids, input_mask, labels=None)       
        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        print("\r%f" % loss, end='')        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
##########################################################################################################################################
#prepare test
all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)
test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)   
#evaluate test
model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []
for input_ids, input_mask, segment_ids,_ in tqdm_notebook(test_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
    preds.append(logits.detach().cpu().numpy().squeeze())
#write
p = []
it = 0
for i in preds:
    for j in i:
        p.append([j,it])
        it += 1        
import csv
with open("Data_Science_HW4.csv", "w") as writer:
    for i in p:
        #print(i)
        w = csv.writer(writer)
        w.writerow(i)
