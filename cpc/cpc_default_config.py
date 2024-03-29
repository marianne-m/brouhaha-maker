# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse


def get_default_cpc_config():
    parser = set_default_cpc_config(argparse.ArgumentParser())
    return parser.parse_args([])


def set_default_cpc_config(parser):
    # Run parameters
    group = parser.add_argument_group('Architecture configuration',
                                      description="The arguments defining the "
                                      "model's architecture.")
    group.add_argument('--hiddenEncoder', type=int, default=256,
                       help='Hidden dimension of the encoder network.')
    group.add_argument('--hiddenGar', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network')
    group.add_argument('--nPredicts', type=int, default=12,
                       help='Number of steps to predict.')
    group.add_argument('--negativeSamplingExt', type=int, default=128,
                       help='Number of negative samples to take.')
    group.add_argument('--learningRate', type=float, default=2e-4)
    group.add_argument('--schedulerStep', type=int, default=-1,
                       help='Step of the learning rate scheduler: at each '
                       'step the learning rate is divided by 2. Default: '
                       'no scheduler.')
    group.add_argument('--schedulerRamp', type=int, default=None,
                       help='Enable a warm up phase for the learning rate: '
                       'adds a linear ramp of the given size.')
    group.add_argument('--beta1', type=float, default=0.9,
                       help='Value of beta1 for the Adam optimizer')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='Value of beta2 for the Adam optimizer')
    group.add_argument('--epsilon', type=float, default=1e-08,
                       help='Value of epsilon for the Adam optimizer')
    group.add_argument('--sizeWindow', type=int, default=20480,
                       help='Number of frames to consider at each batch.')
    group.add_argument('--nEpoch', type=int, default=200,
                       help='Number of epoch to run')
    group.add_argument('--samplingType', type=str, default='samespeaker',
                       choices=['samespeaker', 'uniform',
                                'samesequence', 'sequential'],
                       help='How to sample the negative examples in the '
                       'CPC loss.')
    group.add_argument('--nLevelsPhone', type=int, default=1,
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--typeLevelsPhone', type=str, default="linear",
                       choices=['linear','bi-LSTM'],
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--cpc_mode', type=str, default=None,
                       choices=['reverse', 'bert', 'none'],
                       help='Some variations on CPC.')
    group.add_argument('--encoder_type', type=str,
                       choices=['cpc', 'mfcc', 'lfb'],
                       default='cpc',
                       help='Replace the encoder network by mfcc features '
                       'or learned filter banks')
    group.add_argument('--normMode', type=str, default='layerNorm',
                       choices=['instanceNorm', 'ID', 'layerNorm',
                                'batchNorm'],
                       help="Type of normalization to use in the encoder "
                       "network (default is layerNorm).")
    group.add_argument('--onEncoder', action='store_true',
                       help="(Supervised mode only) Perform the "
                       "classification on the encoder's output.")
    group.add_argument('--random_seed', type=int, default=None,
                       help="Set a specific random seed.")
    group.add_argument('--adversarial', action='store_true',
                       help="(Depreciated) Activate the speaker adversarial "
                       "training.")
    group.add_argument('--speakerEmbedding', type=int, default=0,
                       help="(Depreciated) Feed the prediction network with "
                       "speaker embeddings along with the usual sequence.")
    group.add_argument('--arMode', default='LSTM',
                       choices=['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer'],
                       help="Architecture to use for the auto-regressive "
                       "network (default is lstm).")
    group.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    group.add_argument('--rnnMode', type=str, default='transformer',
                        choices=['transformer', 'RNN', 'LSTM', 'biLSTM', 'linear',
                                 'ffd', 'conv4', 'conv8', 'conv12'],
                       help="Architecture to use for the prediction network")
    group.add_argument('--dropout', action='store_true',
                       help="Add a dropout layer at the output of the "
                       "prediction network.")
    group.add_argument('--abspos', action='store_true',
                       help='If the prediction network is a transformer, '
                       'active to use absolute coordinates.')
    group.add_argument('--clustering', type=str, default=None,
                       choices=['deepEmbedded', 'deepClustering',
                                'CTCClustering'],
                       help="(Research) add a clustering loss on top of the "
                       "current training.")
    group.add_argument('--n_clusters', type=int, default=200,
                       help="(Clustering only) Number of clusters to compute")
    group.add_argument('--cluster_delay', type=int, default=0,
                       help="(Clustering only) wait the given number of "
                       "epoch before activating the clustering loss.")
    group.add_argument('--cluster_iter', type=int, default=100,
                       help="(Clustering only) Maximal number of iterations "
                       "when computing the clusters")
    group.add_argument('--clustering_update', type=str, default='kmean',
                       choices=['kmean', 'dpmean'],
                       help="(Clustering only) Clustering method to use.")
    group.add_argument('--multihead_rnn', action='store_true',
                       help="Use one rnn network with k classifiers on top "
                       "of it instead of k independant rnn networks")
    group.add_argument('--transformer_pruning', type=int, default=0)

    group_augment = parser.add_argument_group('Data augmentation configuration',
                                      description="The arguments defining the "
                                      "data augmentation.")
    group_augment.add_argument('--noise_extension', type=str, default='.wav')
    group_augment.add_argument('--augment_future', action='store_true')
    group_augment.add_argument('--augment_past', action='store_true')
    group_augment.add_argument('--augment_type', type=str,
                                choices=['none', 'bandreject', 'pitch',
                                         'pitch_dropout', 'pitch_quick',
                                         'additive', 'reverb', 'time_dropout',
                                         'reverb_dropout'], nargs='*')
    group_augment.add_argument('--bandreject_scaler', type=float, default=1.0)
    group_augment.add_argument('--additive_noise_snr', type=float, default=15.0)
    group_augment.add_argument('--t_ms', type=int, default=100)
    group_augment.add_argument('--pathDBNoise', type=str, default=None)
    group_augment.add_argument('--pathSeqNoise', type=str, default=None)
    return parser
