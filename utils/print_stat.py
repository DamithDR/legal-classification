from sklearn.metrics import recall_score, precision_score, f1_score


def print_information(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    # labels = list(set(predictions))
    #
    # for label in labels:
    #     print()
    #     print("Stat of the {} Class".format(label))
    #     print("Recall {}".format(
    #         recall_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))
    #     print("Precision {}".format(
    #         precision_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))
    #     print("F1 Score {}".format(
    #         f1_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighted F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    macro_f1 = f1_score(real_values, predictions, average='macro')
    micro_f1 = f1_score(real_values, predictions, average='micro')
    print("Macro F1 Score {}".format(macro_f1))
    print("Micro F1 Score {}".format(micro_f1))
    return macro_f1, micro_f1
