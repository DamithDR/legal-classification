import json
import os

import pandas as pd


def run(directory_path_train, directory_path_test, directory_path_dev, prefix='ECHR'):
    training_files = os.listdir(path=directory_path_train)
    save_path = 'data/processed/echr/'
    training_texts = []
    training_labels = []
    for file in training_files:
        path = directory_path_train + file

        with open(path) as json_data:
            data = json.load(json_data)
            training_texts.append(data['TEXT'])
            if len(data['VIOLATED_ARTICLES']) == 0 and len(data['VIOLATED_PARAGRAPHS']) == 0 and len(
                    data['VIOLATED_BULLETPOINTS']) == 0:
                training_labels.append(0)
            else:
                training_labels.append(1)
    training_df = pd.DataFrame({'text': training_texts, 'labels': training_labels})
    training_df.to_csv(save_path + prefix + '_train.csv', index=False, sep='\t')

    # test set
    test_files = os.listdir(path=directory_path_test)

    test_texts = []
    test_labels = []
    for file in test_files:
        path = directory_path_test + file

        with open(path) as json_data:
            data = json.load(json_data)
            test_texts.append(data['TEXT'])
            if len(data['VIOLATED_ARTICLES']) == 0 and len(data['VIOLATED_PARAGRAPHS']) == 0 and len(
                    data['VIOLATED_BULLETPOINTS']) == 0:
                test_labels.append(0)
            else:
                test_labels.append(1)
    test_df = pd.DataFrame({'text': test_texts, 'labels': test_labels})
    test_df.to_csv(save_path+ prefix + '_test.csv', index=False, sep='\t')

    # dev set
    dev_files = os.listdir(path=directory_path_dev)

    dev_texts = []
    dev_labels = []
    for file in dev_files:
        path = directory_path_dev + file

        with open(path) as json_data:
            data = json.load(json_data)
            dev_texts.append(data['TEXT'])
            if len(data['VIOLATED_ARTICLES']) == 0 and len(data['VIOLATED_PARAGRAPHS']) == 0 and len(
                    data['VIOLATED_BULLETPOINTS']) == 0:
                dev_labels.append(0)
            else:
                dev_labels.append(1)

    dev_df = pd.DataFrame({'text': dev_texts, 'labels': dev_labels})
    dev_df.to_csv(save_path + prefix + '_dev.csv', index=False, sep='\t')


if __name__ == '__main__':
    directory_path_train = 'data/ECHR_Dataset/EN_train/'
    directory_path_test = 'data/ECHR_Dataset/EN_test/'
    directory_path_dev = 'data/ECHR_Dataset/EN_dev/'
    run(directory_path_train,directory_path_test,directory_path_dev,"ECHR")

    directory_path_train = 'data/ECHR_Dataset/EN_train_Anon/'
    directory_path_test = 'data/ECHR_Dataset/EN_test_Anon/'
    directory_path_dev = 'data/ECHR_Dataset/EN_dev_Anon/'
    run(directory_path_train,directory_path_test,directory_path_dev,"ECHR_Anon")
