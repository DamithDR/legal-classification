import argparse
import json
import os

import numpy as np
import pandas as pd
import torch.cuda
from simpletransformers.classification import ClassificationModel
from transformers import BertTokenizer

from fuse.main_model_fuse import ModelLoadingInfo, load_model, fuse_models
from utils.print_stat import print_information

TRAIN_DATA_PATH = 'data/ECHR_Dataset/EN_train/'
TEST_DATA_PATH = 'data/ECHR_Dataset/EN_test/'
DEV_DATA_PATH = 'data/ECHR_Dataset/EN_dev/'

# load training files
training_files = os.listdir(TRAIN_DATA_PATH)
test_files = os.listdir(TEST_DATA_PATH)
dev_files = os.listdir(DEV_DATA_PATH)

training_set = []
test_set = []
dev_set = []


def sep_text_to_regions(df, n_models=3):
    max_token_length = 400

    data_splits = {}
    conclusion_splits = {}
    document_id_splits = {}

    # for text, conclusion in zip(train_df['TEXT'], train_df['CONCLUSION']):
    for text, conclusion, itemid in zip(df['TEXT'], df['CONCLUSION'], df['ITEMID']):
        if 'Inadmissible'.__eq__(conclusion) or 'inadmissible'.__eq__(conclusion):
            conclusion = 0
        else:
            conclusion = 1
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
    parser = argparse.ArgumentParser(
        description='''fuse multiple models ''')
    parser.add_argument('--device_number', required=False, help='cuda device number', default=0)
    parser.add_argument('--no_of_models', required=True, help='no of models to fuse', default=3)
    parser.add_argument('--n_fold', required=True, help='n_fold predictions', default=3)
    arguments = parser.parse_args()
    n_models = int(arguments.no_of_models)
    n_fold = int(arguments.n_fold)

    print(f'***Training Files Loading Started***')
    for file in training_files:
        with open(TRAIN_DATA_PATH + file, encoding='utf-8') as content:
            training_set.append(json.loads(content.read()))
    print(f'---Training Files Loading Finished---')

    print(f'***Dev Files Loading Started***')
    for file in dev_files:
        with open(DEV_DATA_PATH + file, encoding='utf-8') as content:
            dev_set.append(json.loads(content.read()))
    print(f'---Dev Files Loading Finished---')

    print(f'***Test Files Loading Started***')
    for file in test_files:
        with open(TEST_DATA_PATH + file, encoding='utf-8') as content:
            test_set.append(json.loads(content.read()))
    print(f'---Test Files Loading Finished---')

    train_df = pd.DataFrame(training_set)
    test_df = pd.DataFrame(test_set)
    dev_df = pd.DataFrame(dev_set)

    # processed data
    training_text_splits, training_label_splits, training_id_splits = sep_text_to_regions(train_df, n_models)
    test_text_splits, test_label_splits, test_id_splits = sep_text_to_regions(test_df, n_models)
    dev_text_splits, dev_label_splits, dev_id_splits = sep_text_to_regions(dev_df, n_models)

    output_path = 'outputs/'
    fused_model_path = 'outputs/fused_model/'
    train_args = {
        'evaluate_during_training': True,
        'logging_steps': 1000,
        'num_train_epochs': 3,
        'evaluate_during_training_steps': 100,
        'save_eval_checkpoints': False,
        # 'manual_seed': 888888,
        # 'manual_seed': 777777,
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
            train_args['best_model_dir'] = model_path
            model_paths.append(model_path)
            model = ClassificationModel(
                "bert", "nlpaueb/legal-bert-base-uncased", num_labels=2, use_cuda=torch.cuda.is_available(),
                args=train_args
            )

            model.train_model(df_train, eval_df=df_eval)

            model.save_model(output_dir=model_path)
            print('model save')

    # ========================================================================

    # model_paths = ['../../outputs/model_0/', '../../outputs/model_1/', '../../outputs/model_2/']
    # fusing multiple models
    print('model fusing started')
    roberta = ModelLoadingInfo(name="nlpaueb/legal-bert-base-uncased", tokenizer_name="nlpaueb/legal-bert-base-uncased",
                               classification=True)
    models_to_fuse = [ModelLoadingInfo(name=model, tokenizer_name=model, classification=True) for model in model_paths]
    base_model = load_model(roberta)
    fused_model = fuse_models(base_model, models_to_fuse)
    # saving fused model for predictions
    fused_model.save_pretrained(fused_model_path)
    tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    tokenizer.save_pretrained(fused_model_path)
    print('fused model saved')

    print('Starting Predictions')
    macros = []
    micros = []
    for fold in range(0, n_fold):
        # predictions
        print('Starting Prediction fold no : ' + str(fold))

        general_model = ClassificationModel(
            "bert", fused_model_path, use_cuda=torch.cuda.is_available(), args=train_args
        )
        results = []

        for i in range(0, n_models):
            df_test = pd.DataFrame(
                {'text': test_text_splits[i], 'label': test_label_splits[i], 'id': test_id_splits[i]})
            predictions, raw_outputs = general_model.predict(df_test['text'].tolist())
            for id_score, zero_score, one_score in zip(test_id_splits[i], list(raw_outputs[:, 0]),
                                                       list(raw_outputs[:, 1])):
                results.append((id_score, zero_score, one_score))

        score_results = pd.DataFrame(results, columns=['ids', 'scores_0', 'scores_1'])
        final_scores = score_results.groupby(by=['ids']).mean()
        # final_scores = score_results.groupby(by=['ids']).max()

        final_scores.loc[final_scores['scores_0'] <= final_scores['scores_1'], 'prediction'] = 1
        final_scores.loc[final_scores['scores_0'] > final_scores['scores_1'], 'prediction'] = 0

        test_df['gold'] = 1
        test_df.loc[test_df['CONCLUSION'] == 'Inadmissible', 'gold'] = 0
        test_df.loc[test_df['CONCLUSION'] == 'inadmissible', 'gold'] = 0

        gold_answers = []
        for doc_id in final_scores.index:
            ans = test_df.loc[test_df['ITEMID'] == doc_id, 'gold']
            gold_answers.append(list(ans)[0])
        final_scores['gold'] = gold_answers

        macro_f1, micro_f1 = print_information(final_scores, 'prediction', 'gold')
        macros.append(macro_f1)
        micros.append(micro_f1)

    print('Final Results')
    print('=====================================================================')

    macro_str = "Macro F1 Mean - {} | STD - {}\n".format(np.mean(macros), np.std(macros))
    micro_str = "Macro F1 Mean - {} | STD - {}".format(np.mean(micros), np.std(micros))
    print(macro_str)
    print(micro_str)

    print('======================================================================')

    with open('results.txt', 'w') as f:
        f.write(macro_str + micro_str)


if __name__ == '__main__':
    run()
