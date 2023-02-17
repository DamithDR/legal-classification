import argparse

import pandas as pd
import torch
from datasets import load_dataset
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

from utils.print_stat import print_information


def run():
    parser = argparse.ArgumentParser(
        description='''fuse multiple models ''')
    parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
    parser.add_argument('--base_model', required=False, help='n_fold predictions',
                        default='bert-base-cased')

    parser.add_argument('--dataset', required=False, help='dataset for predictions', default='ecthr_cases')
    parser.add_argument('--model_type', required=False, help='type of the model', default='bert')
    arguments = parser.parse_args()
    dataset = arguments.dataset

    print('data loading started')

    # datasets : SetFit/20_newsgroups, ecthr_cases , hyperpartisan_news_detection

    if dataset.__eq__('ecthr_cases'):
        dataset = load_dataset(dataset)
        train_labels = []
        test_labels = []
        dev_labels = []
        for rational in dataset['train']['silver_rationales']:
            if len(rational) > 0:
                train_labels.append(1)
            else:
                train_labels.append(0)
        for rational in dataset['test']['silver_rationales']:
            if len(rational) > 0:
                test_labels.append(1)
            else:
                test_labels.append(0)
        for rational in dataset['validation']['silver_rationales']:
            if len(rational) > 0:
                dev_labels.append(1)
            else:
                dev_labels.append(0)


        train_df = pd.DataFrame({'text': dataset['train']['facts'], 'labels': train_labels})
        train_df, df_finetune = train_test_split(train_df, test_size=0.2)
        test_df = pd.DataFrame({'text': dataset['test']['facts'], 'labels': test_labels})
        dev_df = pd.DataFrame(
            {'text': dataset['validation']['facts'], 'labels': dev_labels})

    elif dataset.__eq__('hyperpartisan_news_detection'):
        dataset = load_dataset(dataset, 'bypublisher')
        train_df = pd.DataFrame(
            {'text': dataset.data['train']['text'], 'labels': dataset.data['train']["hyperpartisan"]})
        train_df, df_finetune = train_test_split(train_df, test_size=0.2, random_state=777)
        df_finetune, dev_df = train_test_split(df_finetune, test_size=0.5, random_state=777)
        test_df = pd.DataFrame(
            {'text': dataset.data['validation']['text'], 'labels': dataset.data['validation']["hyperpartisan"]})
        print('hyperpartisan_news_detection')
    elif dataset.__eq__('20_news_categories'):
        dataset = pd.read_csv('data/processed/20news/20news.csv', sep='\t')
        train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=777)
        train_df, df_finetune = train_test_split(train_df, test_size=0.2, random_state=777)
        df_finetune, dev_df = train_test_split(df_finetune, test_size=0.5, random_state=777)
    elif dataset.__eq__('ECHR'):
        train_df = pd.read_csv('data/processed/echr/ECHR_train.csv', sep='\t')
        test_df = pd.read_csv('data/processed/echr/ECHR_test.csv', sep='\t')
        dev_df = pd.read_csv('data/processed/echr/ECHR_dev.csv', sep='\t')
        dev_df, df_finetune = train_test_split(dev_df, test_size=0.2, random_state=777)
    elif dataset.__eq__('ECHR_Anon'):
        train_df = pd.read_csv('data/processed/echr/ECHR_Anon_train.csv', sep='\t')
        test_df = pd.read_csv('data/processed/echr/ECHR_Anon_test.csv', sep='\t')
        dev_df = pd.read_csv('data/processed/echr/ECHR_Anon_dev.csv', sep='\t')
        dev_df, df_finetune = train_test_split(dev_df, test_size=0.2, random_state=777)

    train_args = {
        'evaluate_during_training': True,
        'logging_steps': 1000,
        'num_train_epochs': 3,
        'evaluate_during_training_steps': 100,
        'save_eval_checkpoints': False,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        # 'manual_seed': 888888,
        # 'manual_seed': 777,
        'train_batch_size': 32,
        'eval_batch_size': 8,
        'overwrite_output_dir': True,
    }

    label_set = set(train_df['labels'].tolist())
    label_to_num_dict = {}
    num_to_label_dict = {}
    index = 0
    for lab in label_set:
        label_to_num_dict[lab] = index
        num_to_label_dict[index] = lab
        index += 1

    # encoding
    train_labels = []
    test_labels = []
    dev_labels = []
    finetune_labels = []
    for train_label in train_df['labels']:
        train_labels.append(label_to_num_dict.get(train_label))
    # for test_label in test_df['labels']:
    #     test_labels.append(label_to_num_dict.get(test_label))
    for dev_label in dev_df['labels']:
        dev_labels.append(label_to_num_dict.get(dev_label))
    for finetune_label in df_finetune['labels']:
        finetune_labels.append(label_to_num_dict.get(finetune_label))
    train_df['labels'] = train_labels
    # test_df['labels'] = test_labels
    dev_df['labels'] = dev_labels
    df_finetune['labels'] = finetune_labels
    print('data loading and encoding finished starting data chunking')

    # training data preparation
    print('training started')

    model = ClassificationModel(
        arguments.model_type, arguments.base_model, use_cuda=torch.cuda.is_available(),
        args=train_args,
        num_labels=len(label_set)
    )

    model.train_model(train_df, eval_df=dev_df)

    predictions, raw_outputs = model.predict(test_df['text'].tolist())

    decoded_predictions = []
    for prediction in predictions:
        decoded_predictions.append(num_to_label_dict.get(prediction))

    test_df['predictions']= decoded_predictions

    print_information(test_df, 'predictions', 'labels')







if __name__ == '__main__':
    run()
