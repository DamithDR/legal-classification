import argparse
import json
import logging
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch.cuda
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from fuse.main_model_fuse import ModelLoadingInfo, load_model, fuse_models
from utils.print_stat import print_information
from datasets import load_dataset

import torch.multiprocessing

def sep_text_to_regions(df, n_models=3):
    max_token_length = 400

    data_splits = {}
    conclusion_splits = {}
    document_id_splits = {}

    # for text, conclusion in zip(train_df['TEXT'], train_df['CONCLUSION']):
    for text, conclusion, itemid in zip(df['text'], df['labels'], df.index):
        text = " ".join(text).split(' ')
        text_splits = np.array_split(text, n_models)
        for i in range(0, n_models):
            split = text_splits[i]
            split_len = len(split)
            if split_len > max_token_length:
                arr_split_size = (split_len // max_token_length) + 1

                sub_texts = np.array_split(split, arr_split_size)
                conclusions = [conclusion] * len(sub_texts)
                document_ids = [itemid] * len(sub_texts)

                # all these sub texts go to one model
                if data_splits.__contains__(i):
                    for st in sub_texts:
                        data_splits[i].append(' '.join(st.tolist()))
                else:
                    lst = []
                    for st in sub_texts:
                        lst.append(' '.join(st.tolist()))
                    data_splits[i] = lst
                if conclusion_splits.__contains__(i):
                    conclusion_splits[i] = conclusion_splits[i] + conclusions
                else:
                    conclusion_splits[i] = conclusions
                if document_id_splits.__contains__(i):
                    document_id_splits[i] = document_id_splits[i] + document_ids
                else:
                    document_id_splits[i] = document_ids
            else:
                if data_splits.__contains__(i):
                    data_splits[i].append(' '.join(split))
                else:
                    data_splits[i] = [' '.join(split)]
                if conclusion_splits.__contains__(i):
                    conclusion_splits[i].append(conclusion)
                else:
                    conclusion_splits[i] = [conclusion]
                if document_id_splits.__contains__(i):
                    document_id_splits[i].append(itemid)
                else:
                    document_id_splits[i] = [itemid]

    return data_splits, conclusion_splits, document_id_splits


def run():
    logging.basicConfig(filename='3m777r-results.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='''fuse multiple models ''')
    parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
    parser.add_argument('--no_of_models', required=True, help='no of models to fuse', default=3)
    parser.add_argument('--n_fold', required=True, help='n_fold predictions', default=3)
    parser.add_argument('--base_model', required=False, help='n_fold predictions',
                        default='bert-base-cased')

    parser.add_argument('--dataset', required=False, help='dataset for predictions', default='ecthr_cases')
    parser.add_argument('--epochs', required=False, help='num_train_epochs', default=3)
    parser.add_argument('--model_type', required=False, help='type of the model', default='bert')
    arguments = parser.parse_args()
    n_models = int(arguments.no_of_models)
    n_fold = int(arguments.n_fold)
    dataset = arguments.dataset
    train_epochs = int(arguments.epoches)

    print('data loading started')

    if dataset.__eq__('20_news_categories'):
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
    elif dataset.__eq__('case-2021'):
        train_df = pd.read_json('data/processed/case-2021/train.json')
        train_df = train_df.rename(columns={'label': 'labels'})
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=777)
        train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=777)
        dev_df, df_finetune = train_test_split(dev_df, test_size=0.5, random_state=777)

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
    dev_labels = []
    finetune_labels = []
    for train_label in train_df['labels']:
        train_labels.append(label_to_num_dict.get(train_label))
    for dev_label in dev_df['labels']:
        dev_labels.append(label_to_num_dict.get(dev_label))
    for finetune_label in df_finetune['labels']:
        finetune_labels.append(label_to_num_dict.get(finetune_label))
    train_df['labels'] = train_labels
    dev_df['labels'] = dev_labels
    df_finetune['labels'] = finetune_labels
    print('data loading and encoding finished starting data chunking')

    # processed data
    training_text_splits, training_label_splits, training_id_splits = sep_text_to_regions(train_df, n_models)
    test_text_splits, test_label_splits, test_id_splits = sep_text_to_regions(test_df, n_models)
    dev_text_splits, dev_label_splits, dev_id_splits = sep_text_to_regions(dev_df, n_models)
    finetune_text_splits, finetune_label_splits, finetune_id_splits = sep_text_to_regions(df_finetune, n_models)

    print('data chunking finished')

    output_path = 'outputs/'
    fused_model_path = 'outputs/fused_model/'
    fused_finetuned_model_path = 'outputs/fused_finetuned_model/'
    train_args = {
        'evaluate_during_training': True,
        'logging_steps': 1000,
        'num_train_epochs': train_epochs,
        'evaluate_during_training_steps': 100,
        'save_eval_checkpoints': False,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        'train_batch_size': 32,
        'eval_batch_size': 8,
        'overwrite_output_dir': True,
        'output_dir': output_path

    }

    if torch.cuda.is_available():
        torch.device('cuda')
        torch.cuda.set_device(int(arguments.device_number))
    # ========================================================================
    # training data preparation
    print('training started')
    model_paths = []
    for i in range(0, n_models):
        model_path = output_path + 'model_' + str(i)
        df_train = pd.DataFrame({'text': training_text_splits[i], 'labels': training_label_splits[i]})
        df_eval = pd.DataFrame({'text': dev_text_splits[i], 'labels': dev_label_splits[i]})
        # full_df = pd.concat([df_train, df_eval])
        # df_train, df_eval = train_test_split(full_df, test_size=0.2, random_state=777)
        train_args['best_model_dir'] = model_path
        model_paths.append(model_path)
        model = ClassificationModel(
            arguments.model_type, arguments.base_model, use_cuda=torch.cuda.is_available(),
            args=train_args,
            num_labels=len(label_set)
        )

        model.train_model(df_train, eval_df=df_eval)

        print('model saved')
    print('split models saving finished')

    # ========================================================================
    # # fusing multiple models
    print('model fusing started')
    roberta = ModelLoadingInfo(name=arguments.base_model, tokenizer_name=arguments.base_model,
                               classification=True)
    models_to_fuse = [ModelLoadingInfo(name=model, tokenizer_name=model, classification=True) for model in model_paths]
    base_model = load_model(roberta)
    fused_model = fuse_models(base_model, models_to_fuse)
    # saving fused model for predictions
    fused_model.save_pretrained(fused_model_path)
    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)
    tokenizer.save_pretrained(fused_model_path)
    print('fused model saved')

    # load the saved model
    train_args['best_model_dir'] = fused_finetuned_model_path
    train_args['learning_rate'] = 1e-04

    general_model = ClassificationModel(
        arguments.model_type, fused_model_path, use_cuda=torch.cuda.is_available(), args=train_args,
        num_labels=len(label_set)
    )

    df_eval = pd.DataFrame()
    df_finetune_training = pd.DataFrame()
    # further fine tuning - this step is important
    for i in range(0, n_models):
        training_chunk = pd.DataFrame({'text': finetune_text_splits[i], 'labels': finetune_label_splits[i]})
        eval = pd.DataFrame({'text': dev_text_splits[i], 'labels': dev_label_splits[i]})
        df_eval = pd.concat([df_eval, eval])
        df_finetune_training = pd.concat([df_finetune_training, training_chunk])

    general_model.train_model(df_finetune_training, eval_df=df_eval)
    general_model.save_model(output_dir=fused_finetuned_model_path)

    fine_tuned_model = general_model  # to use directly

    print('Starting Predictions')
    macros = []
    micros = []

    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            for fold in range(0, n_fold):
                # predictions
                print('Starting Prediction fold no : ' + str(fold))

                results = []

                for i in range(0, n_models):
                    df_test = pd.DataFrame(
                        {'text': test_text_splits[i], 'labels': test_label_splits[i], 'id': test_id_splits[i]})
                    predictions, raw_outputs = fine_tuned_model.predict(df_test['text'].tolist())
                    probabilities = softmax(raw_outputs, axis=1)
                    for id_score, probs in zip(test_id_splits[i], list(probabilities)):
                        results.append((id_score, *probs))
                column_name_list = ['ids']
                for i in range(0, len(label_set)):
                    column_name_list.append(str(i))
                score_results = pd.DataFrame(results, columns=column_name_list)
                final_scores = score_results.groupby(by=['ids']).mean()

                prediction_columns = final_scores.idxmax(axis=1)

                prediction_columns = list(prediction_columns.astype('int32'))
                decoded_predictions = []
                for pred in prediction_columns:
                    decoded_predictions.append(num_to_label_dict.get(pred))

                final_scores['prediction'] = decoded_predictions  # set the column prediction
                gold_answers = []
                for doc_id in final_scores.index:
                    ans = test_df.loc[test_df.index == doc_id, 'labels']
                    gold_answers.append(list(ans)[0])
                final_scores['gold'] = gold_answers

                macro_f1, micro_f1 = print_information(final_scores, 'prediction', 'gold')
                macros.append(macro_f1)
                micros.append(micro_f1)

            print('Final Results')
            print('=====================================================================')

            macro_str = "Macro F1 Mean - {} | STD - {}\n".format(np.mean(macros), np.std(macros))
            micro_str = "Micro F1 Mean - {} | STD - {}".format(np.mean(micros), np.std(micros))
            print(macro_str)
            print(micro_str)

            print('======================================================================')

            print(macro_str + micro_str)
    print("Done")


if __name__ == '__main__':
    run()
