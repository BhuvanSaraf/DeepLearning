import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_to_idx = {}
    for i, lab in enumerate(labels):
        label_to_idx[lab] = i

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, labels


def print_confusion_matrix(cm, labels):
    header = "      " + "  ".join(f"P{lab}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = "  ".join(f"{v:4d}" for v in cm[i])
        print(f"T{lab}   {row}")


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


def per_class_precision_recall_f1(cm, labels):

    n = len(labels)
    precision = {}
    recall = {}
    f1 = {}

    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp   
        fn = np.sum(cm[i, :]) - tp   
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision[lab] = p
        recall[lab] = r
        f1[lab] = f

    return precision, recall, f1


def macro_average(per_class_dict):
    values = list(per_class_dict.values())
    return sum(values) / len(values)


def micro_average_precision_recall_f1(cm):

    n = cm.shape[0]
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for i in range(n):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn

    p_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0

    return p_micro, r_micro, f1_micro


def print_all_metrics(y_true, y_pred, labels=None):

    cm, labels = confusion_matrix(y_true, y_pred, labels)
    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = per_class_precision_recall_f1(cm, labels)

    print("Confusion matrix:")
    print_confusion_matrix(cm, labels)
    print()

    print(f"Accuracy: {acc:.4f}")
    print()

    print("Per class metrics:")
    print("      precision  recall    f1")
    for lab in labels:
        print(f"{lab:>4}   {precision[lab]:.4f}     {recall[lab]:.4f}    {f1[lab]:.4f}")
    print()

    macro_p = macro_average(precision)
    macro_r = macro_average(recall)
    macro_f1 = macro_average(f1)
    print(f"Macro average -> precision: {macro_p:.4f}  recall: {macro_r:.4f}  f1: {macro_f1:.4f}")

    micro_p, micro_r, micro_f1 = micro_average_precision_recall_f1(cm)
    print(f"Micro average -> precision: {micro_p:.4f}  recall: {micro_r:.4f}  f1: {micro_f1:.4f}")

    return {
        "confusion_matrix": cm,
        "labels": labels,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def percent_rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    r = rmse(y_true, y_pred)
    span = y_true.max() - y_true.min()
    if span == 0:
        return float('nan')
    return 100 * r / span


if __name__ == '__main__':
    yt = [0, 0, 1, 1, 2, 2, 2]
    yp = [0, 1, 1, 1, 2, 2, 0]
    print_all_metrics(yt, yp)

    print()
    yt_r = [1.0, 2.0, 3.0, 4.0]
    yp_r = [1.1, 1.9, 3.2, 3.7]
    print('rmse:', rmse(yt_r, yp_r))
    print('%rmse:', percent_rmse(yt_r, yp_r))
