import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from config import *

softmax = nn.Softmax(dim=2)


# aggregate results of VAD per frame for overlapping sliding windows
def aggregate_results_vad_overlap(
        proba, pred, label, overlap, seq_size, batch_size):
    len_agg = int((proba.shape[1] + (overlap - 1) * seq_size) / overlap)
    proba_agg = torch.zeros((8, len_agg, 2)).cpu()
    pred_agg = torch.zeros((8, len_agg, 2)).cpu()
    label_agg = torch.zeros((8, len_agg)).int().cpu()
    count = torch.zeros((8, len_agg, 2)).cpu()

    for i in tqdm(range(0, int(proba.shape[1] / seq_size))):
        offset = int(seq_size * i / overlap)
        proba_agg[:,
                  offset: offset + seq_size,
                  :] = proba_agg[:,
                                 offset: offset + seq_size,
                                 :] + proba[:,
                                            seq_size * i: (i + 1) * seq_size,
                                            :]
        pred_agg[:,
                 offset: offset + seq_size,
                 :] = pred_agg[:,
                               offset: offset + seq_size,
                               :] + pred[:,
                                         seq_size * i: (i + 1) * seq_size,
                                         :]
        label_agg[:,
                  offset: offset + seq_size] = label_agg[:,
                                                         offset: offset + seq_size] + label[:,
                                                                                            seq_size * i: (i + 1) * seq_size]
        count[:, offset: offset + seq_size, :] = count[:, offset: offset +
                                                       seq_size, :] + torch.ones((batch_size, seq_size, 2))

    proba_agg = torch.div(proba_agg, count.cpu())
    pred_agg = torch.div(pred_agg, count.cpu())
    label_agg = torch.div(label_agg, count[:, :, 0].cpu())

    return proba_agg, pred_agg, label_agg


# aggregate a vector for a given frame
def vector_agg_i(i, vec, label, overlap, position):
    weight_overlap = 1 / overlap

    vec_agg = vec[:, i] + \
        2 * weight_overlap * (label[:, i - 1] == label[:, i]).int() * vec[:, i - 1] + \
        2 * weight_overlap * (label[:, i + 1] ==
                              label[:, i]).int() * vec[:, i + 1]
    if position != 'start':
        vec_agg += weight_overlap * \
            (label[:, i - 2] == label[:, i]).int() * vec[:, i - 2]
    if position != 'end':
        vec_agg += weight_overlap * \
            (label[:, i + 2] == label[:, i]).int() * vec[:, i + 2]

    return vec_agg

# aggregate weights for a given frame


def weight_agg_i(i, label, overlap, position):
    weight_overlap = 1 / overlap

    weight_agg = 1 + \
        2 * weight_overlap * (label[:, i - 1] == label[:, i]).int() + \
        2 * weight_overlap * (label[:, i + 1] == label[:, i]).int()
    if position != 'start':
        weight_agg += weight_overlap * (label[:, i - 2] == label[:, i]).int()
    if position != 'end':
        weight_agg += weight_overlap * (label[:, i + 2] == label[:, i]).int()

    return weight_agg


# aggregate results of SNR/reverb predictions per frame for overlapping
# sliding windows
def aggregate_results_snr_reverb_overlap(pred, label, overlap):
    len_agg = int(pred.shape[1] / overlap)
    pred_agg = torch.zeros((8, len_agg)).cpu()
    label_agg = torch.zeros((8, len_agg)).cpu()

    for i in tqdm(range(1, pred.shape[1] - overlap, overlap)):
        if i == 1:
            pred_agg[:, int(i / overlap)] = vector_agg_i(i,
                                                         pred, label, overlap, 'start')
            weight_agg = weight_agg_i(i, label, overlap, 'start')

        elif pred.shape[1] - i < overlap + 2:
            pred_agg[:, int(i / overlap)] = vector_agg_i(i,
                                                         pred, label, overlap, 'end')
            weight_agg = weight_agg_i(i, label, overlap, 'end')

        else:
            pred_agg[:, int(i / overlap)] = vector_agg_i(i,
                                                         pred, label, overlap, 'middle')
            weight_agg = weight_agg_i(i, label, overlap, 'middle')

        pred_agg[:, int(i / overlap)] = torch.div(pred_agg[:,
                                                           int(i / overlap)], weight_agg)
        label_agg[:, int(i / overlap)] = label[:, i]

    pred_agg = pred_agg.view(-1)
    label_agg = label_agg.view(-1)
    return pred_agg, label_agg


# compute scores for VAD systems
def compute_scores_vad(proba, label):
    auc_score = np.round(metrics.roc_auc_score(label, proba[:, 1]), 3)
    f1_score = np.round(metrics.f1_score(label, proba.argmax(1)), 2)
    _, fp, fn, _ = np.round(metrics.confusion_matrix(
        label, proba.argmax(1), normalize='true').ravel() * 100, 1)
    return auc_score, f1_score, fp, fn


# compute the Mean Absolute Percentage Error for SNR and reverb predictions
def compute_mape_loss_snr_reverb(label_agg, pred_agg, values, delta):
    # average MAPE
    loss_avg = float(MAPE(label_agg, pred_agg))

    # MAPE per bucket of REVERB gold values
    losses = [loss_avg]
    for val in values:
        if val == values[0]:
            position = 'start'
        elif val == values[-1]:
            position = 'end'
        else:
            position = 'middle'

        label = extract_values_bucket(
            label_agg, label_agg, val, delta, position)
        pred = extract_values_bucket(pred_agg, label_agg, val, delta, position)

        if pred.shape[0] != 0:
            loss = float(MAPE(label, pred))
            losses += [loss]
        else:
            losses += [0]

    return losses


# print MAPE scores
def print_mape_scores(dataset, val_type, losses, bucket_values, delta):
    print(f"MAPE loss for {val_type} predictions on {dataset}")
    print(f"Average loss = {losses[0]:.2f}")
    print(f"Loss < {bucket_values[0] + delta} = {losses[1]:.2f}")
    print(
        f"Loss {bucket_values[1] - delta} - {bucket_values[1] + delta} = {losses[2]:.2f}")
    print(
        f"Loss {bucket_values[2] - delta} - {bucket_values[2] + delta} = {losses[3]:.2f}")
    print(
        f"Loss {bucket_values[3] - delta} - {bucket_values[3] + delta} = {losses[4]:.2f}")
    print(f"Loss > {bucket_values[4] - delta} = {losses[5]:.2f}")


# compute the correlation between SNR/reverb predictions and VAD probabilities
def compute_corr_with_vad(proba_vad, label, pred):
    pred_proba = proba_vad.max(dim=1)[0].view(-1, 128).mean(dim=1)

    # correlation between gold and predicted reverb values
    corr_gold_val_pred_val = np.corrcoef(label, pred)[0][1]
    # correlation between gold reverb values and probabilities of predicted
    # VAD labels
    corr_gold_val_pred_proba = np.corrcoef(label, pred_proba)[0][1]
    # correlation between predicted reverb values and probabilities of
    # predicted VAD labels
    corr_pred_val_pred_proba = np.corrcoef(pred, pred_proba)[0][1]

    return corr_gold_val_pred_val, corr_gold_val_pred_proba, corr_pred_val_pred_proba


# print correlation scores
def print_score_corr(dataset, val_type, corr_gold_val_pred_val,
                     corr_gold_val_pred_proba, corr_pred_val_pred_proba):
    print(f"Correlation with {val_type} predictions on {dataset}")
    print(
        f"Correlation between gold and predicted {val_type} = {corr_gold_val_pred_val:.2f}")
    print(
        f"Correlation between gold {val_type} and probabilities of predicted labels = {corr_gold_val_pred_proba:.2f}")
    print(
        f"Correlation between predicted {val_type} and probabilities of predicted labels = {corr_pred_val_pred_proba:.2f}")


# compute accuracy scores for SNR and reverb predictions
def compute_accuracy(pred_agg, label_agg, values, delta):
    pred_agg_round = torch.round(pred_agg * 10 / 10)

    # average accuracy
    acc_avg = float(
        (torch.round(pred_agg) == torch.round(label_agg)).double().mean())

    # accuracy per bucket of gold reverb values
    accuracies = [acc_avg]
    acc_bucket_avg = 0
    count_total = 0
    for val in values:
        if val == values[0]:
            position = 'start'
            count = (label_agg < val + delta).sum()
            count_total += count
        elif val == values[-1]:
            position = 'end'
            count = (label_agg >= (val - delta)).sum()
            count_total += count
        else:
            position == 'middle'
            count = (
                (label_agg >= (
                    val -
                    delta)) & (
                    label_agg < (
                        val +
                        delta))).sum()
            count_total += count

        pred = extract_values_bucket(
            pred_agg_round, label_agg, val, delta, position)
        acc = float(
            ((pred >= val -
              delta) & (
                pred < val +
                delta)).double().mean())
        if np.isnan(acc):
            acc = 0
        accuracies += [acc]
        acc_bucket_avg += acc * count

    acc_bucket_avg = float(acc_bucket_avg / len(label_agg))
    accuracies = [accuracies[0]] + [acc_bucket_avg] + accuracies[1:]
    return accuracies


# print accuracy scores
def print_acc_scores(dataset, val_type, accuracies, bucket_values, delta):
    print(f"Accuracy for {val_type} predictions on {dataset}")
    print(f"Average accuracy = {accuracies[0]:.2%}")
    print(f"Average accuracy per bucket = {accuracies[1]:.2%}")
    print(f"Accuracy < {bucket_values[0] + delta} = {accuracies[2]:.2%}")
    print(
        f"Accuracy {bucket_values[1] - delta} - {bucket_values[1] + delta} = {accuracies[3]:.2%}")
    print(
        f"Accuracy {bucket_values[2] - delta} - {bucket_values[2] + delta} = {accuracies[4]:.2%}")
    print(
        f"Accuracy {bucket_values[3] - delta} - {bucket_values[3] + delta} = {accuracies[5]:.2%}")
    print(f"Accuracy > {bucket_values[4] - delta} = {accuracies[6]:.2%}")


# extract values of a vector in a given bucket of label values
# (value-delta, value+delta)
def extract_values_bucket(vec, label, value, delta, position):
    if position == 'start':
        return vec[label < value + delta]
    elif position == 'middle':
        return vec[(label >= value - delta) & (label < value + delta)]
    elif position == 'end':
        return vec[label >= value - delta]


# print VAD scores
def print_vad_scores(dataset, model, auc_score, f1_score,
                     fp, fn, sliding_window=None):
    print(f"Performance of {model} for VAD on {dataset}:")
    if sliding_window == 'proba':
        print("Results by frame obtained by averaging class probabilities")
    elif sliding_window == 'pred':
        print("Results by frame obtained by averaging class scores")
    print(f"AUC score: {auc_score}")
    print(f"F1 score: {f1_score}")
    print(f"FP rate: {fp} % and FN rate: {fn} %")


# save scores
def save_score(score, dataset, comment, score_file):
    output_score_file = open(score_file, 'a')
    time = datetime.now()
    line = ";".join([str(time), dataset] + score + [comment])
    output_score_file.write(line + "\n")
    output_score_file.close()
    print(f"Score saved at {score_file}")
