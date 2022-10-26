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
from simpletransformers.config.model_args import T5Args
from simpletransformers.t5 import T5Model
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from fuse.main_model_fuse_t5 import ModelLoadingInfoT5, load_model, fuse_models
from utils.print_stat import print_information
from datasets import load_dataset

import torch.multiprocessing

# otherwise the shared memory is not enough so it throws an error
torch.multiprocessing.set_sharing_strategy('file_system')


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
    TASK_NAME = 'binary classification'

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
                        default='t5-base')
    arguments = parser.parse_args()
    n_models = int(arguments.no_of_models)
    n_fold = int(arguments.n_fold)

    print('data loading started')

    dataset = load_dataset("ecthr_cases")
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
    dev_df = pd.DataFrame({'text': dataset['validation']['facts'], 'labels': dev_labels})

    print('data loading finished starting data chunking')

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
        "overwrite_output_dir": True,
        'output_dir': output_path,
        "max_seq_length": 512,
        "eval_batch_size": 32,
        "use_multiprocessing": False,
        "num_beams": None,
        "do_sample": True,
        "max_length": 50,
        "top_k": 50,
        "top_p": 0.95,
        # "num_return_sequences": 3,
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'save_eval_checkpoints': False,
        'evaluate_during_training': True,
    }

    # train_args = {
    #     'evaluate_during_training': True,
    #     'logging_steps': 1000,
    #     'num_train_epochs': 5,
    #     'evaluate_during_training_steps': 100,
    #     'save_eval_checkpoints': False,
    #     # 'manual_seed': 888888,
    #     # 'manual_seed': 777,
    #     'train_batch_size': 32,
    #     'eval_batch_size': 8,
    #     'overwrite_output_dir': True,
    #     'output_dir': output_path
    #
    # }

    if torch.cuda.is_available():
        torch.device('cuda')
        torch.cuda.set_device(int(arguments.device_number))
    # ========================================================================
    # training data preparation
    # print('training started')
    # model_paths = []
    # for i in range(0, n_models):
    #     model_path = output_path + 'model_' + str(i)
    #     df_train = pd.DataFrame({'input_text': training_text_splits[i], 'target_text': training_label_splits[i]})
    #     df_train['target_text'] = df_train['target_text'].apply(str)
    #     df_eval = pd.DataFrame({'input_text': dev_text_splits[i], 'target_text': dev_label_splits[i]})
    #     df_eval['target_text'] = df_eval['target_text'].apply(str)
    #     df_train['prefix'] = TASK_NAME
    #     df_eval['prefix'] = TASK_NAME
    #     # full_df = pd.concat([df_train, df_eval])
    #     # df_train, df_eval = train_test_split(full_df, test_size=0.2, random_state=777)
    #     train_args['best_model_dir'] = model_path
    #     model_paths.append(model_path)
    #     # training_df = pd.DataFrame()
    #     # development_df = pd.DataFrame()
    #     # for train_sample, train_label in zip(training_text_splits[i], training_label_splits[i]):
    #     #     training_df.append([TASK_NAME, train_sample, train_label])
    #     # for dev_sample, dev_label in zip(dev_text_splits[i], dev_label_splits[i]):
    #     #     development_df.append([TASK_NAME, dev_sample, dev_label])
    #
    #     model = T5Model(
    #         "t5",
    #         arguments.base_model,
    #         use_cuda=torch.cuda.is_available(),
    #         args=train_args
    #     )
    #
    #     model.train_model(df_train, eval_data=df_eval)
    #
    #     model.save_model(output_dir=model_path)
    #     print('model saved')
    # print('split models saving finished')
    #
    # # ========================================================================
    #
    model_paths = ['outputs/model_0/', 'outputs/model_1/', 'outputs/model_2/']
    # # fusing multiple models
    print('model fusing started')
    model_load = ModelLoadingInfoT5(name=arguments.base_model, tokenizer_name=arguments.base_model,
                                  classification=True)
    models_to_fuse = [ModelLoadingInfoT5(name=model, tokenizer_name=model, classification=True) for model in model_paths]
    base_model = load_model(model_load)
    fused_model = fuse_models(base_model, models_to_fuse)
    # saving fused model for predictions
    fused_model.save_pretrained(fused_model_path)
    tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)
    tokenizer.save_pretrained(fused_model_path)
    print('fused model saved')

    # load the saved model
    train_args['best_model_dir'] = fused_finetuned_model_path
    train_args['learning_rate'] = 1e-04

    general_model = T5Model(
        "t5",
        fused_model_path,
        use_cuda=torch.cuda.is_available(),
        args=train_args
    )

    df_eval = pd.DataFrame()
    df_finetune_training = pd.DataFrame()

    # further fine tuning - this step is important
    for i in range(0, n_models):
        training_chunk = pd.DataFrame({'input_text': finetune_text_splits[i], 'target_text': finetune_label_splits[i]})
        eval_chunk = pd.DataFrame({'input_text': dev_text_splits[i], 'target_text': dev_label_splits[i]})
        df_finetune_training = pd.concat([df_finetune_training, training_chunk])
        df_eval = pd.concat([df_eval, eval_chunk])

    df_finetune_training['prefix'] = TASK_NAME
    df_eval['prefix'] = TASK_NAME
    df_finetune_training['target_text'] = df_finetune_training['target_text'].apply(str)
    df_eval['target_text'] = df_eval['target_text'].apply(str)

    general_model.train_model(df_finetune_training, eval_data=df_eval)
    general_model.save_model(output_dir=fused_finetuned_model_path)

    fine_tuned_model = general_model  # to use directly
    #
    # fine_tuned_model = ClassificationModel(
    #     "bert", fused_finetuned_model_path, use_cuda=torch.cuda.is_available(), args=train_args
    # )

    print('Starting Predictions')
    macros = []
    micros = []

    # with open('out.txt', 'w') as f:
    #     with redirect_stdout(f):
    for fold in range(0, n_fold):
        # predictions
        print('Starting Prediction fold no : ' + str(fold))

        results = []
        for i in range(0, n_models):
            test_list = []
            for test_sample, test_label in zip(test_text_splits[i], test_label_splits[i]):
                test_list.append([TASK_NAME + ":" + test_sample])

            raw_outputs = fine_tuned_model.predict(test_list)
            # probabilities = softmax(raw_outputs, axis=1)

            # for id_score, zero_score, one_score in zip(test_id_splits[i], list(probabilities[:, 0]),
            #                                            list(probabilities[:, 1])):
            #     results.append((id_score, zero_score, one_score))
            for id_score, prediction in zip(test_id_splits[i], raw_outputs):
                try:
                    results.append((id_score, int(prediction)))
                except:
                    print("wrong output cannot convert to int " + prediction)
                    print("An exception occurred")

        score_results = pd.DataFrame(results, columns=['ids', 'prediction'])
        # final_scores = score_results.groupby(by=['ids']).mean()
        final_scores = score_results.groupby(by=['ids']).max()

        # final_scores.loc[final_scores['scores_0'] <= final_scores['scores_1'], 'prediction'] = 1
        # final_scores.loc[final_scores['scores_0'] > final_scores['scores_1'], 'prediction'] = 0

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
