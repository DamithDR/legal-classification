import os

import pandas as pd
from pandas import DataFrame

file_path = 'D:\DataSets\\20news-18828\\'

lst = os.listdir(path=file_path)

dataset_file = 'data/processed/20news.csv'

dataset = DataFrame()

content_list = []
label_list = []

with open(dataset_file, 'a', encoding='UTF-8') as df:
    df.write('text\tlabels')
for path in lst:
    folder_path = file_path + path
    print(f'working on file path = {folder_path}')
    files_list = os.listdir(path=folder_path)
    for file in files_list:
        with open(folder_path + '\\' + file, 'r', encoding='cp1252') as f:
            content = f.read()
            content = content.replace('\n', ' ')
            content = content.replace('\t', ' ')

            content_list.append(content)
            label_list.append(path)

dataset['text'] = content_list
dataset['labels'] = label_list

dataset.to_csv(dataset_file, sep='\t')

print('done')
