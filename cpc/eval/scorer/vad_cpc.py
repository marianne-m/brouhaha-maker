import os
import torch
from tqdm import tqdm
from config import *
from utils.misc_score import compute_scores_vad, save_score, aggregate_results_vad_overlap, softmax
from dataset_maker.utils.misc import findAllSeqs_relativePath


# compute and save scores of a CPC model for VAD
class ScorerVadCpc:

    def __init__(self, dataset, path_pred, overlap, batch_size, seq_size):
        self.dataset = dataset                        # dataset name
        # number of predictions per frame when using sliding windows
        # (default=1)
        self.overlap = overlap
        self.batch_size = batch_size                  # size of prediction batches
        self.seq_size = seq_size                      # size of prediction sequences
        # path to the directory with CPC outputs
        self.path_pred = path_pred
        self.pred = torch.empty(
            (self.batch_size, 0, 2)).cuda()       # CPC predictions
        self.label = torch.empty(
            (self.batch_size, 0)).int().cuda()  # gold VAD labels
        # CPC probabilities (predictions converted into probabilities)
        self.proba = None

    # load files with VAD outputs

    def __load_files_overlap(self):
        files = findAllSeqs_relativePath(
            self.path_pred,
            extension='proba.pt',
            start="speech",
            load_cache=False)
        for file in tqdm(files):
            self.pred = torch.cat(
                (self.pred, torch.load(
                    os.path.join(
                        self.path_pred, file)).view(
                    self.batch_size, self.seq_size, -1)), dim=1)
            self.label = torch.cat(
                (self.label, torch.load(
                    os.path.join(
                        self.path_pred, file.replace(
                            'proba', 'gold'))).int().view(
                    self.batch_size, -1)), dim=1)

        self.pred = self.pred.cpu()
        self.label = self.label.int().cpu()

    # compute VAD scores

    def __call__(self, comment):
        self.__load_files_overlap()
        # convert predictions into probabilities
        self.proba = softmax(self.pred)

        # no overlap
        if self.overlap == 1:
            auc_score, f1_score, fp, fn = compute_scores_vad(
                self.proba.view(-1, 2), self.label.view(-1).int())
            print_vad_scores(self.dataset, "CPC", auc_score, f1_score, fp, fn)
            save_score(list(map(str, [auc_score, f1_score, fp, fn])),
                       self.dataset, comment, PATH_OUTPUT_SCORE_VAD)

        # overlapping sliding windows
        else:
            proba_agg, pred_agg, label_agg = aggregate_results_vad_overlap(
                self.proba, self.pred, self.label, self.overlap, self.seq_size, self.batch_size)

            auc_score, f1_score, fp, fn = compute_scores_vad(
                proba_agg.view(-1, 2), label_agg.view(-1).int())
            print_vad_scores(
                self.dataset,
                "CPC",
                auc_score,
                f1_score,
                fp,
                fn,
                sliding_window='proba')
            save_score(list(map(str,
                                [auc_score,
                                 f1_score,
                                 fp,
                                 fn])),
                       self.dataset,
                       f'{comment}_agg_proba',
                       PATH_OUTPUT_SCORE_VAD)

            auc_score, f1_score, fp, fn = compute_scores_vad(
                pred_agg.view(-1, 2), label_agg.view(-1).int())
            print_vad_scores(
                self.dataset,
                "CPC",
                auc_score,
                f1_score,
                fp,
                fn,
                sliding_window='pred')
            save_score(list(map(str,
                                [auc_score,
                                 f1_score,
                                 fp,
                                 fn])),
                       self.dataset,
                       f'{comment}_agg_pred',
                       PATH_OUTPUT_SCORE_VAD)
