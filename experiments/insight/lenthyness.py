import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description='''fuse multiple models ''')
parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
parser.add_argument('--base_model', required=False, help='n_fold predictions',
                    default='bert-base-cased')

parser.add_argument('--dataset', required=False, help='dataset for predictions', default='ecthr_cases')

arguments = parser.parse_args()
dataset = arguments.dataset

if dataset.__eq__('20_news_categories'):
    train_df = pd.read_csv('data/processed/20news/20news.csv', sep='\t')
elif dataset.__eq__('ECHR'):
    train_df = pd.read_csv('data/processed/echr/ECHR_train.csv', sep='\t')
elif dataset.__eq__('ECHR_Anon'):
    train_df = pd.read_csv('data/processed/echr/ECHR_Anon_train.csv', sep='\t')
elif dataset.__eq__('case-2021'):
    train_df = pd.read_json('data/processed/case-2021/train.json')

total_cases = 0
cases_more_than_4096 = 0
cases_between_512_4096 = 0
cases_below_512 = 0
text_list = train_df['text'].tolist()
for text in text_list:
    if len(text.split(' ')) > 4096:
        cases_more_than_4096 += 1
    elif len(text.split(' ')) > 512:
        cases_between_512_4096 += 1
    else:
        cases_below_512 += 1
    total_cases+=1

print(f'Below 512 : {(cases_below_512/total_cases)*100}')
print(f'Between 512 and 4096 : {(cases_between_512_4096/total_cases)*100}')
print(f'Above 4096 : {(cases_more_than_4096/total_cases)*100}')