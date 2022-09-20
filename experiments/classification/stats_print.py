import operator
import os
import json

import numpy as np
import pandas as pd

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

# print(f'***Test Files Loading Started***')
# for file in test_files:
#     with open(TEST_DATA_PATH + file, encoding='utf-8') as content:
#         test_set.append(json.loads(content.read()))
# print(f'---Test Files Loading Finished---')
#
# print(f'***Dev Files Loading Started***')
# for file in dev_files:
#     with open(DEV_DATA_PATH + file, encoding='utf-8') as content:
#         dev_set.append(json.loads(content.read()))
# print(f'---Dev Files Loading Finished---')

train_df = pd.DataFrame(training_set)
# test_df = pd.DataFrame(test_set)
# dev_df = pd.DataFrame(dev_set)

v_sum = 0
n_sum = 0
total_violations = 0
for row in training_set:
    articles = row['VIOLATED_ARTICLES']
    paras = row['VIOLATED_PARAGRAPHS']
    bullets = row['VIOLATED_BULLETPOINTS']

    if len(articles) == 0 or len(paras) == 0 or len(bullets) == 0:

        n_sum += 1
    else:
        v_sum += 1



print("violated {}".format(v_sum))
print("non violated {}".format(n_sum))

# print('total cases train ' + str(len(train_df)))

# violations = 0
# non_violations = 0
# max_tokens = 0
# no_of_tokens = []
# docs_greater_than_5000 = 0
# docs_greater_than_2000 = 0
# chunk_size = 510
# chunks_dict = {}
#
# total_violations = 0
# total_non_violations = 0
# for text, conclusion, articles, in zip(train_df['TEXT'], train_df['CONCLUSION']):
#     # meta data of dataset
#     if 'inadmissible'.__eq__(conclusion.lower()):
#         non_violations += 1
#     else:
#         violations += 1
#
#     words = ' '.join(text).split(' ')
#     no_of_tokens.append(len(words))
#     if len(words) > 5000:
#         docs_greater_than_5000 += 1
#     if len(words) > 2000:
#         docs_greater_than_2000 += 1
#     if max_tokens < len(words):
#         max_tokens = len(words)
#
#     no_of_chunks = len(words) // 510 + 1
#     if chunks_dict.__contains__(no_of_chunks):
#         chunks_dict[no_of_chunks] = chunks_dict[no_of_chunks] + 1
#     else:
#         chunks_dict[no_of_chunks] = 1
#
#     words = ' '.join(text).split(' ')
#
# print('docs_greater_than_2000 ' + str(docs_greater_than_2000))
# print('docs_greater_than_5000 ' + str(docs_greater_than_5000))
# print('maximum no of tokens ' + str(max_tokens))
# print('violations = ' + str(violations))
# print('non violations = ' + str(non_violations))
# print('chunks dict')
# print(sorted(chunks_dict.items(), key=operator.itemgetter(0)))
