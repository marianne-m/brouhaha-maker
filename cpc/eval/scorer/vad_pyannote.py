import torch
from tqdm import tqdm
from config import *
from utils.misc_score import compute_scores_vad, save_score
from dataset_maker.utils.misc import findAllSeqs_relativePath

# compute and save scores of a pyannote model for VAD


class ScorerVadPya:

    def __init__(self, dataset, path_proba):
        self.dataset = dataset              # dataset name
        # path to the directory with pyannote output probabilities
        self.path_proba = path_proba
        self.label = []                     # gold VAD labels
        self.proba = []                     # pyannote probabilities

    # load files with gold values and output probabilities

    def __load_files(self):
        files = findAllSeqs_relativePath(self.path_proba, extension='.txt')

        for file in tqdm(files):
            with open(file, 'r') as f_proba:
                proba_tmp = f_proba.readlines()
            proba_tmp = [[float(x.split()[0]), float(x.split()[1])]
                         for x in proba_tmp]

            with open(file.replace('proba', 'gold'), 'r') as f_label:
                label_tmp = f_label.readlines()
            label_tmp = [float(x) for x in label_tmp]

            if len(label_tmp) < len(proba_tmp):
                label_tmp += [0] * (len(proba_tmp) - len(label_tmp))
            else:
                label_tmp = label_tmp[:len(proba_tmp)]

            self.proba += proba_tmp
            self.label += label_tmp

        self.proba = torch.FloatTensor(self.proba).cpu()
        self.label = torch.IntTensor(self.label).cpu()

    # compute VAD scores

    def __call__(self, comment):
        self.__load_files()
        auc_score, f1_score, fp, fn = compute_scores_vad(
            self.proba, self.label)
        print_vad_scores(self.dataset, "pyannote", auc_score, f1_score, fp, fn)
        save_score(map(str, [auc_score, f1_score, fp, fn]),
                   self.dataset, comment, PATH_OUTPUT_SCORE_VAD)
