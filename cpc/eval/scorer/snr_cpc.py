import os
import torch
import numpy as np
from tqdm import tqdm
from config import *
from utils.misc_score import aggregate_results_snr_reverb_overlap, save_score, compute_corr_with_vad, compute_mape_loss_snr_reverb, compute_accuracy, softmax, print_mape_scores, print_score_corr, print_acc_scores
from dataset_maker.utils.misc import findAllSeqs_relativePath


# compute and save score of a CPC model for SNR prediction
class ScorerSnrCpc:

    def __init__(self, dataset, path_pred, overlap, batch_size, seq_size):
        self.dataset = dataset                          # dataset name
        # path to the directory with CPC outputs
        self.path_pred = path_pred
        # number of predictions per frame when using sliding windows
        # (default=1)
        self.overlap = overlap
        self.batch_size = batch_size                    # size of prediction batches
        self.seq_size = seq_size                        # size of prediction sequences
        self.pred = torch.empty((self.batch_size, 0)).float(
        ).cuda()         # CPC predictions of SNR values
        self.label = torch.empty(
            (self.batch_size, 0)).float().cuda()        # gold SNR labels
        self.pred_vad = torch.empty(
            (self.batch_size, 0, 2)).float().cuda()   # CPC predictions of VAD
        self.label_vad = torch.empty(
            (self.batch_size, 0)).int().cuda()      # gold VAD labels
        # CPC probabilities of VAD (predictions converted into probabilities)
        self.proba_vad = None

    # load files with CPC outputs for SNR predictions and VAD

    def __load_files(self):
        files = findAllSeqs_relativePath(
            self.path_pred,
            extension='gold.pt',
            start="snr",
            load_cache=False)
        for file in tqdm(files):
            self.pred = torch.cat(
                (self.pred, torch.load(
                    os.path.join(
                        self.path_pred, file.replace(
                            'gold', 'pred'))).float().view(
                    self.batch_size, -1)), dim=1)
            self.label = torch.cat(
                (self.label, torch.load(
                    os.path.join(
                        self.path_pred, file)).float().view(
                    self.batch_size, -1)), dim=1)

        files = findAllSeqs_relativePath(
            self.path_pred,
            extension='proba.pt',
            start="speech",
            load_cache=False)
        for file in tqdm(files):
            self.pred_vad = torch.cat(
                (self.pred_vad, torch.load(
                    os.path.join(
                        self.path_pred, file)).view(
                    self.batch_size, self.seq_size, -1)), dim=1)
            self.label_vad = torch.cat(
                (self.label_vad, torch.load(
                    os.path.join(
                        self.path_pred, file.replace(
                            'proba', 'gold'))).int().view(
                    self.batch_size, -1)), dim=1)
        self.proba_vad = softmax(self.pred_vad)

        self.pred = self.pred.cpu()
        self.label = self.label.cpu()
        self.pred_vad = self.pred_vad.cpu()
        self.label_vad = self.label_vad.cpu()
        self.proba_vad = self.proba_vad.cpu()

    # compute Mean Absolute Percentage Error (MAPE) for SNR prediction

    def __mape_loss(self):
        # average MAPE
        losses = compute_mape_loss_snr_reverb(
            self.label_agg, self.pred_agg, SNR_VALUES, SNR_DELTA)
        print_mape_scores(self.dataset, "SNR", losses, SNR_VALUES, SNR_DELTA)
        return losses

    # compute the correlation between SNR values and probabilities of
    # predicted VAD labels

    def __corr_snr(self):
        corr_gold_snr_pred_snr, corr_gold_snr_pred_proba, corr_pred_snr_pred_proba = compute_corr_with_vad(
            self.proba_vad, self.label, self.pred)
        print_score_corr(
            self.dataset,
            "SNR",
            corr_gold_snr_pred_snr,
            corr_gold_snr_pred_proba,
            corr_pred_snr_pred_proba)
        return [corr_gold_snr_pred_snr,
                corr_gold_snr_pred_proba, corr_pred_snr_pred_proba]

    # compute the accuracy of SNR predictions (log scale)

    def __accuracy(self):
        accuracies = compute_accuracy(
            self.pred_agg.log(),
            self.label_agg.log(),
            SNR_LOG_VALUES,
            SNR_LOG_DELTA)
        print_acc_scores(
            self.dataset,
            "SNR",
            accuracies,
            SNR_VALUES,
            SNR_DELTA)
        return accuracies

    # compute SNR scores

    def __call__(self, comment):
        self.__load_files()

        if self.overlap != 1:
            self.pred_agg, self.label_agg = aggregate_results_snr_reverb_overlap(
                self.pred, self.label, self.overlap)
        else:
            self.pred_agg = self.pred.view(-1)
            self.label_agg = self.label.view(-1)

        self.pred = self.pred.view(-1)
        self.label = self.label.view(-1)
        self.proba_vad = self.proba_vad.view(-1, 2)
        self.pred_vad = self.pred_vad.view(-1, 2)
        self.label_vad = self.label_vad.view(-1, 2)

        # MAPE loss
        list_loss = self.__mape_loss()
        list_loss = [str(np.round(x, 2)) for x in list_loss]

        # correlation with VAD probabilities
        list_corr = self.__corr_snr()
        list_corr = [str(np.round(x, 2)) for x in list_corr]

        # accuracy
        list_acc = self.__accuracy()
        list_acc = [str(np.round(x * 100, 2)) for x in list_acc]

        save_score(
            list_loss +
            list_corr +
            list_acc,
            self.dataset,
            comment,
            PATH_OUTPUT_SCORE_SNR)
