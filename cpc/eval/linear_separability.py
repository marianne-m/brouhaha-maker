# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
import random

import cpc.criterion as cr
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
from cpc.data_augmentation import augmentation_factory
from cpc.train import loadCriterion, loadCriterionSNR, loadCriterionReverb
from cpc.cpc_default_config import set_default_cpc_config


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_step(feature_maker, criterion_speech, criterion_snr, criterion_reverb, data_loader, optimizer, snr_weight, reverb_weight):

    if feature_maker.optimize:
        feature_maker.train()
    criterion_speech.train()
    if criterion_snr:
        criterion_snr.train()
    if criterion_reverb:
        criterion_reverb.train()

    logs = {"locLoss_train": 0,  "locAcc_train": 0, "locLoss_speech": 0, "locLoss_snr": 0, "locLoss_reverb": 0}

    for step, fulldata in enumerate(data_loader):

        optimizer.zero_grad()
        batch_data, label_speech, label_snr, label_reverb = fulldata
        batch_data,_ = batch_data[:, 0, ...], batch_data[:, 1, ...]
        c_feature, encoded_data, _ = feature_maker(batch_data, None)
        if not feature_maker.optimize:
            c_feature, encoded_data = c_feature.detach(), encoded_data.detach()
        all_losses_speech, all_acc_speech = criterion_speech(batch_data, c_feature, encoded_data, label_speech)
        totLoss = all_losses_speech.sum()
        logs["locLoss_train"] += np.asarray([all_losses_speech.mean().item()])
        logs["locLoss_speech"] += np.asarray([all_losses_speech.mean().item()])
        logs["locAcc_train"] += np.asarray([all_acc_speech.mean().item()])

        if criterion_snr:
            all_losses_snr, all_acc_snr = criterion_snr(c_feature, encoded_data, label_snr)
            totLoss += all_losses_snr.sum() * snr_weight
            logs["locLoss_train"] += np.asarray([all_losses_snr.mean().item()]) * snr_weight
            logs["locLoss_snr"] += np.asarray([all_losses_snr.mean().item()])
            logs["locAcc_train"] += np.asarray([all_acc_snr.mean().item()])

        if criterion_reverb:
            all_losses_reverb, all_acc_reverb = criterion_reverb(c_feature, encoded_data, label_reverb)
            totLoss += all_losses_reverb.sum() * reverb_weight
            logs["locLoss_train"] += np.asarray([all_losses_reverb.mean().item()]) * reverb_weight
            logs["locLoss_reverb"] += np.asarray([all_losses_reverb.mean().item()])
            logs["locAcc_train"] += np.asarray([all_acc_reverb.mean().item()])

        #totLoss = torch.Tensor([totLoss]).view(1,-1)
        totLoss.backward()
        optimizer.step()

    logs = utils.update_logs(logs, step)
    logs["iter"] = step

    return logs


def val_step(feature_maker, criterion_speech, criterion_snr, criterion_reverb, data_loader, snr_weight, reverb_weight, path_predictions=None):

    feature_maker.eval()
    criterion_speech.eval()
    if criterion_snr:
        criterion_snr.eval()
    if criterion_reverb:
        criterion_reverb.eval()

    logs = {"locLoss_val": 0,  "locAcc_val": 0}
    count = 0

    for step, fulldata in enumerate(data_loader):
        with torch.no_grad():
            batch_data, label_speech, label_snr, label_reverb = fulldata
            batch_data,_ = batch_data[:, 0, ...], batch_data[:, 1, ...]
            c_feature, encoded_data, _ = feature_maker(batch_data, None)
            all_losses_speech, all_acc_speech = criterion_speech(batch_data, c_feature, encoded_data, label_speech, count=count, path_predictions=path_predictions)
            logs["locLoss_val"] += np.asarray([all_losses_speech.mean().item()])
            logs["locAcc_val"] += np.asarray([all_acc_speech.mean().item()])
            print(criterion_snr)
            print(criterion_reverb)
            if criterion_snr:
                all_losses_snr, all_acc_snr = criterion_snr(c_feature, encoded_data, label_snr, count=count, path_predictions=path_predictions)
                logs["locLoss_val"] += np.asarray([all_losses_snr.mean().item()]) * snr_weight
                logs["locAcc_val"] += np.asarray([all_acc_snr.mean().item()])

            if criterion_reverb:
                all_losses_reverb, all_acc_reverb = criterion_reverb(c_feature, encoded_data, label_reverb, count=count, path_predictions=path_predictions)
                logs["locLoss_val"] += np.asarray([all_losses_reverb.mean().item()]) * reverb_weight
                logs["locAcc_val"] += np.asarray([all_acc_reverb.mean().item()])

            count += 1
    logs = utils.update_logs(logs, step)
    return logs


def run(feature_maker,
        criterion_speech,
        criterion_snr,
        criterion_reverb,
        train_loader,
        val_loader,
        optimizer,
        logs,
        n_epochs,
        path_checkpoint,
        snr_weight,
        reverb_weight):

    start_epoch = len(logs["epoch"])
    best_acc = -1

    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):

        logs_train = train_step(feature_maker, criterion_speech, criterion_snr, criterion_reverb, train_loader,
                                optimizer, snr_weight, reverb_weight)
        logs_val = val_step(feature_maker, criterion_speech, criterion_snr, criterion_reverb, val_loader, snr_weight, reverb_weight)
        print('')
        print('_'*50)
        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')
        print(logs_train)
        print(logs_val)
        utils.show_logs("Training loss", logs_train)
        utils.show_logs("Validation loss", logs_val)
        print('_'*50)
        print('')

        if logs_val["locAcc_val"] > best_acc:
            best_state = deepcopy(fl.get_module(feature_maker).state_dict())
            best_acc = logs_val["locAcc_val"]

        logs["epoch"].append(epoch)
        for key, value in dict(logs_train, **logs_val).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        if (epoch % logs["saveStep"] == 0 and epoch > 0) or epoch == n_epochs - 1:
            model_state_dict = fl.get_module(feature_maker).state_dict()
            criterion_speech_state_dict = fl.get_module(criterion_speech).state_dict()
            if criterion_snr:
                criterion_snr_state_dict = fl.get_module(criterion_snr).state_dict()
            else:
                criterion_snr_state_dict = None
            if criterion_reverb:
                criterion_reverb_state_dict = fl.get_module(criterion_reverb).state_dict()
            else:
                criterion_reverb_state_dict = None

            fl.save_checkpoint(model_state_dict, criterion_speech_state_dict,
                               criterion_snr_state_dict, criterion_reverb_state_dict, 
                               optimizer.state_dict(), best_state, 
                               f"{path_checkpoint}_{epoch}.pt")
            utils.save_logs(logs, f"{path_checkpoint}_logs.json")


def eval(feature_maker,
        criterion_speech,
        criterion_snr,
        criterion_reverb,
        val_loader,
        snr_weight,
        reverb_weight,
        path_predictions=None):

    start_time = time.time()
  
    logs_val = val_step(feature_maker, criterion_speech, criterion_snr, criterion_reverb, val_loader, snr_weight, reverb_weight, path_predictions=path_predictions)
    print('')
    print('_'*50)
    print('Ran evaluation '
              f'in {time.time() - start_time:.2f} seconds')
    utils.show_logs("Evaluation loss", logs_val)
    print('_'*50)
    print('')



def parse_args(argv):
    parser = argparse.ArgumentParser(description='Linear separability trainer'
                                     ' (default test in speaker separability)')
    parser = set_default_cpc_config(parser)
    parser.add_argument('--pathDB', type=str,
                        help="Path to the directory containing the audio data.")
    parser.add_argument('--file_extension', type=str, default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--pathTrain', type=str,
                        help="Path to the list of the training sequences.")
    parser.add_argument('--pathVal', type=str,
                        help="Path to the list of the test sequences.")
    parser.add_argument('--load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels. If given, will"
                        " compute the phone separability.")
    parser.add_argument('--pathSnr', type=str, default=None,
                        help="Path to the snr labels. If given, will"
                        " predict snr values.")
    parser.add_argument('--snrWeight', type=float, default=0.01,
                        help="Weight of the SNR loss for training")
    parser.add_argument('--pathReverb', type=str, default=None,
                        help="Path to the reverb labels. If given, will"
                        " predict reverb values (C50).")
    parser.add_argument('--reverbWeight', type=float, default=0.001,
                        help="Weight of the Reverb loss for training")
    parser.add_argument('--pathCheckpoint', type=str, default='out',
                        help="Path of the output directory where the "
                        " checkpoints should be dumped.")
    parser.add_argument('--pathPredictions', type=str, default=None,
                        help="Path of the output directory where the "
                        " predictions should be saved.")
    parser.add_argument('--path_data_split', type=str, default=None,
                        help="Path where to save data split of CPC.")
    parser.add_argument('--save_predictions', action='store_true',
                        help="If activated, save predictions.")
    
    parser.add_argument('--CTC', action='store_true',
                        help="Use the CTC loss (for phone separability only)")
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['train','eval'],
                        help="Training or evaluation mode.")
    parser.add_argument('--unfrozen', action='store_true',
                        help="If activated, update the feature network as well"
                        " as the linear classifier")
    parser.add_argument('--no_pretraining', action='store_true',
                        help="If activated, work from an untrained model.")
    parser.add_argument('--load_criterion', action='store_true',
                        help="If activated, load pretrained criterions.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--size_window', type=int, default=20480, 
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--overlap', type=int, default=1, 
                        help="Number of predictions made for each frame:"
                        "window_overlap = size_window / overlap.")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')

    parser.add_argument('--nGPU', type=int, default=-1,
                        help='Bumber of GPU. Default=-1, use all available '
                        'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")    
    parser.add_argument('--save_step', type=int, default=-1,
                        help="Frequency at which a checkpoint should be saved,"
                        " et to -1 (default) to save only the best checkpoint.")

    args = parser.parse_args(argv)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    if args.save_step <= 0:
        args.save_step = args.n_epoch

    args.load = [str(Path(x).resolve()) for x in args.load]
    args.pathCheckpoint = str(Path(args.pathCheckpoint).resolve())

    return args


def main(argv):

    args = parse_args(argv)
    if args.mode == 'train':
        print("Training mode...")
    elif args.mode == 'eval':
        print("Evaluation mode...")
    else:
        print("Unknown running mode")
        return

    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    load_criterion = args.load_criterion

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache,
                                     speaker_level=0)
    model, hidden_gar, hidden_encoder = fl.loadModel(args.load,
                                                     loadStateDict=not args.no_pretraining)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    dim_features = hidden_encoder if args.get_encoded else hidden_gar

    # Now the criterion
    batch_size = args.batchSizeGPU * args.nGPU
    if not load_criterion:
        phone_labels = None
        snr_labels = None
        reverb_labels = None
        if args.pathPhone is not None:
            phone_labels, n_phones, snr_labels, reverb_labels = parseSeqLabels(args.pathPhone, args.pathSnr, args.pathReverb)
            if not args.CTC:
                print(f"Running phone separability with aligned phones")
                criterion_speech = cr.PhoneCriterion(dim_features, n_phones.item(), args.get_encoded,
                                                     args.typeLevelsPhone, args.nLevelsPhone)
            else:
                print(f"Running phone separability with CTC loss")
                criterion_speech = cr.CTCPhoneCriterion(dim_features,
                                                n_phones.item(), args.get_encoded)
        else:
            print(f"Running speaker separability")
            criterion_speech = cr.SpeakerCriterion(dim_features, len(speakers))

        if args.pathSnr:
            criterion_snr = cr.SNRCriterion(int(args.size_window/snr_labels['step']), dim_features, args.get_encoded)
        else:
            criterion_snr = None
        if args.pathReverb:
            criterion_reverb = cr.ReverbCriterion(int(args.size_window/reverb_labels['step']), dim_features, args.get_encoded)
        else:
            criterion_reverb = None

    else:
        if args.pathPhone is not None:
            print("Loading existing criterions...")
            phone_labels, n_phones, snr_labels, reverb_labels = parseSeqLabels(args.pathPhone, args.pathSnr, args.pathReverb)
            criterion_speech = loadCriterion(args, args.load[0], 160,  # model.getDownsamplingFactor()
                                     len(speakers), n_phones.item())
        elif args.mode == 'eval':
            print("Loading existing criterions...")
            phone_labels = None
            snr_labels = None
            reverb_labels = None
            criterion_speech = loadCriterion(args, args.load[0], 160,  # model.getDownsamplingFactor()
                                     len(speakers), 2)
        else:
            print("Unable to find n_phones")
            print(f"Running speaker separability")
            phone_labels = None
            snr_labels = None
            reverb_labels = None
            criterion_speech = cr.SpeakerCriterion(dim_features, len(speakers))

        if args.pathSnr or args.mode == 'eval':
            criterion_snr = loadCriterionSNR(args, args.load[0])
        else:
            criterion_snr = None
        if args.pathReverb or args.mode == 'eval':
            criterion_reverb = loadCriterionReverb(args, args.load[0])
        else:
            criterion_reverb = None

    if criterion_speech:
        criterion_speech.cuda()
        criterion_speech = torch.nn.DataParallel(criterion_speech, device_ids=range(args.nGPU))
    if criterion_snr:
        criterion_snr.cuda()
        criterion_snr = torch.nn.DataParallel(criterion_snr, device_ids=range(args.nGPU))
    if criterion_reverb:
        criterion_reverb.cuda()
        criterion_reverb = torch.nn.DataParallel(criterion_reverb, device_ids=range(args.nGPU))  

    # Dataset
    if args.path_data_split:
        with open(args.path_data_split, 'w') as f:
            f.write("file start end snr_gold reverb_gold\n")

    if args.mode == 'train':
        if args.pathTrain is not None:
            seq_train = filterSeqs(args.pathTrain, seqNames)
        else:
            seq_train = seqNames
        random.Random(4).shuffle(seq_train)

    if args.pathVal is None:
        print('No validation data specified!')
        seq_val = []
    else:
        seq_val = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seq_train = seq_train[:1000]
        seq_val = seq_val[:100]

    if args.mode == 'train':
        db_train = AudioBatchData(args.pathDB, args.size_window, seq_train,
                                phone_labels, snr_labels, 
                                reverb_labels, len(speakers), 
                                args.path_data_split,
                                augment_future=args.augment_future,
                                augment_past=args.augment_past,
                                augmentation=augmentation_factory(args))
        train_loader = db_train.getDataLoader(batch_size, "uniform", True,
                                          numWorkers=0)

    db_val = AudioBatchData(args.pathDB, args.size_window, seq_val,
                            phone_labels, snr_labels, 
                            reverb_labels, len(speakers), 
                            args.path_data_split,
                            augment_future=args.augment_future,
                            augment_past=args.augment_past,
                            augmentation=augmentation_factory(args))
    val_loader = db_val.getDataLoader(batch_size, 'sequential', False,
                                      numWorkers=0)
    
    # Optimizer
    if args.mode == 'train':
        g_params = list(criterion_speech.parameters())
        if criterion_snr:
            g_params += list(criterion_snr.parameters())
        if criterion_reverb:
            g_params += list(criterion_reverb.parameters())
        model.optimize = False
        model.eval()
        if args.unfrozen:
            print("Working in full fine-tune mode")
            g_params += list(model.parameters())
            model.optimize = True
        else:
            print("Working with frozen features")
            for g in model.parameters():
                g.requires_grad = False

        optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                    betas=(args.beta1, args.beta2),
                                    eps=args.epsilon)

        # Checkpoint directory
        args.pathCheckpoint = Path(args.pathCheckpoint)
        args.pathCheckpoint.mkdir(exist_ok=True)
        args.pathCheckpoint = str(args.pathCheckpoint / "checkpoint")

        with open(f"{args.pathCheckpoint}_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

        run(model, criterion_speech, criterion_snr, criterion_reverb, train_loader, val_loader, optimizer, logs,
            args.n_epoch, args.pathCheckpoint, args.snrWeight, args.reverbWeight)
        
    elif args.mode == 'eval':
        if args.save_predictions:
            print(f"Will save predictions in {args.pathPredictions}")
            eval(model, criterion_speech, criterion_snr, criterion_reverb, val_loader, 
                args.snrWeight, args.reverbWeight, path_predictions=args.pathPredictions)
        else:
            eval(model, criterion_speech, criterion_snr, criterion_reverb, val_loader, 
                args.snrWeight, args.reverbWeight)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
