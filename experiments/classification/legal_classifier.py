import json
import os

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

TRAIN_DATA_PATH = '../../data/ECHR_Dataset/EN_train/'
TEST_DATA_PATH = '../../data/ECHR_Dataset/EN_test/'
DEV_DATA_PATH = '../../data/ECHR_Dataset/EN_dev/'

# load training files
training_files = os.listdir(TRAIN_DATA_PATH)
test_files = os.listdir(TEST_DATA_PATH)
dev_files = os.listdir(DEV_DATA_PATH)

training_set = []
test_set = []
dev_set = []

print(f'***Training Files Loading Started***')
for file in training_files:
    with open(TRAIN_DATA_PATH + file, encoding='utf-8') as content:
        training_set.append(json.loads(content.read()))
print(f'---Training Files Loading Finished---')

print(f'***Test Files Loading Started***')
for file in test_files:
    with open(TEST_DATA_PATH + file, encoding='utf-8') as content:
        test_set.append(json.loads(content.read()))
print(f'---Test Files Loading Finished---')

print(f'***Dev Files Loading Started***')
for file in dev_files:
    with open(DEV_DATA_PATH + file, encoding='utf-8') as content:
        dev_set.append(json.loads(content.read()))
print(f'---Dev Files Loading Finished---')

train_df = pd.DataFrame(training_set)
test_df = pd.DataFrame(test_set)
dev_df = pd.DataFrame(dev_set)

config = BertConfig.from_pretrained('ProsusAI/finbert', output_hidden_states=True, output_attentions=True)
# model = BertForSequenceClassification.from_pretrained('nlpaueb/legal-bert-small-uncased', num_labels=10
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert',config=config)


tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

max_length = 512
# # Training
for text in train_df['TEXT']:
    sequence = " ".join(text)
    tokens = tokenizer.encode_plus(
        sequence,
        add_special_tokens=False,
        return_tensors='pt'
    )
    token_input_ids_chunks = list(tokens['input_ids'][0].split(max_length - 2))
    token_attention_masks_chunks = list(tokens['attention_mask'][0].split(max_length - 2))

    for i in range(len(token_input_ids_chunks)):
        token_input_ids_chunks[i] = torch.cat([torch.tensor([101]), token_input_ids_chunks[i], torch.tensor([102])])
        token_attention_masks_chunks[i] = torch.cat(
            [torch.tensor([1]), token_attention_masks_chunks[i], torch.tensor([1])])

        padding = max_length - len(token_input_ids_chunks[i])
        if padding > 0:
            token_input_ids_chunks[i] = torch.cat([token_input_ids_chunks[i], torch.tensor([0] * padding)])
            token_attention_masks_chunks[i] = torch.cat([token_attention_masks_chunks[i], torch.tensor([0] * padding)])

    input_ids = torch.stack(token_input_ids_chunks)
    attention_mask = torch.stack(token_attention_masks_chunks)
    print('a')

    input_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    outputs = model(**input_dict)
    print(model.state_dict())
