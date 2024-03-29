# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import random
import time
import tqdm
import torch
import numpy as np
import statistics
from pathlib import Path
from copy import deepcopy
from typing import List, Union
from torch.multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.utils.data.sampler import Sampler, BatchSampler

import torchaudio


class AudioBatchData(Dataset):
    def __init__(
        self,
        path,
        sizeWindow,
        seqNames,
        phoneLabelsDict,
        nSpeakers,
        nProcessLoader=10,
        MAX_SIZE_LOADED=4000000000,
        transform=None,
        augment_past=False,
        augment_future=False,
        augmentation=None,
        snrDict=None, 
        reverbDict=None
    ):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabelsDict (dictionnary): if not None, a dictionnary with the
                                             following entries

                                             "step": size of a labelled window
                                             "$SEQ_NAME": list of phonem labels for
                                             the sequence $SEQ_NAME
            - snrDict (dictionnary): if not None, a dictionnary with the
                                     following entries

                                     "step": size of a labelled window
                                     "$SEQ_NAME": snr label for
                                     the sequence $SEQ_NAME
            - reverbDict (dictionnary): if not None, a dictionnary with the
                                        following entries

                                        "step": size of a labelled window
                                        "$SEQ_NAME": reverb label for
                                        the sequence $SEQ_NAME
            - nSpeakers (int): number of speakers to expect.
            - nProcessLoader (int): number of processes to call when loading the
                                    data from the disk
            - MAX_SIZE_LOADED (int): target maximal size of the floating array
                                    containing all loaded data.
        """
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.nProcessLoader = nProcessLoader
        self.dbPath = Path(path)
        self.sizeWindow = sizeWindow
        self.seqNames = [(s, self.dbPath / x) for s, x in seqNames]
        self.reload_pool = Pool(nProcessLoader)
        self.transform = transform

        self.prepare()
        self.speakers = list(range(nSpeakers))
        self.data = []

        self.phoneSize = 0 if phoneLabelsDict is None else phoneLabelsDict["step"]
        self.phoneStep = (
            0 if phoneLabelsDict is None else self.sizeWindow // self.phoneSize
        )

        self.snrSize = 0 if snrDict is None else snrDict["step"]
        self.snrStep = 0 if snrDict is None else self.sizeWindow // self.snrSize

        self.reverbSize = 0 if reverbDict is None else reverbDict["step"]
        self.reverbStep = (
            0 if reverbDict is None else self.sizeWindow // self.reverbSize
        )

        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.snrDict = deepcopy(snrDict)
        self.reverbDict = deepcopy(reverbDict)
        self.loadNextPack(first=True)
        self.loadNextPack()
        self.doubleLabels = False

        self.augment_past = augment_past
        self.augment_future = augment_future
        self.augmentation = augmentation

    def resetPhoneLabels(self, newPhoneLabels, step):
        self.phoneSize = step
        self.phoneStep = self.sizeWindow // self.phoneSize
        self.phoneLabelsDict = deepcopy(newPhoneLabels)
        self.loadNextPack()

    def resetSnr(self, newSnr, step):
        self.snrSize = step
        self.snrStep = self.sizeWindow // self.snrSize
        self.snrDict = deepcopy(newSnr)
        self.loadNextPack()

    def resetReverb(self, newReverb, step):
        self.reverbSize = step
        self.reverbStep = self.sizeWindow // self.reverbSize
        self.reverbDict = deepcopy(newReverb)
        self.loadNextPack()

    def splitSeqTags(seqName):
        path = os.path.normpath(seqName)
        return path.split(os.sep)

    def getSeqNames(self):
        return [str(x[1]) for x in self.seqNames]

    def clear(self):
        if "data" in self.__dict__:
            del self.data
        if "speakerLabel" in self.__dict__:
            del self.speakerLabel
        if "phoneLabels" in self.__dict__:
            del self.phoneLabels
        if "snrLabels" in self.__dict__:
            del self.snrLabels
        if "reverbLabels" in self.__dict__:
            del self.reverbLabels
        if "seqLabel" in self.__dict__:
            del self.seqLabel

    def prepare(self):
        random.shuffle(self.seqNames)
        start_time = time.time()

        print("Checking length...")
        allLength = self.reload_pool.map(extractLength, self.seqNames)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, length in tqdm.tqdm(enumerate(allLength)):
            packageSize += length
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.seqNames)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(
            f"Scanned {len(self.seqNames)} sequences "
            f"in {time.time() - start_time:.2f} seconds"
        )
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    def getNPacks(self):
        return len(self.packageIndex)

    def loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print("Joining pool")
            self.r.wait()
            print(f"Joined process, elapsed={time.time()-start_time:.3f} secs")
            self.nextData = self.r.get()
            self.parseNextDataBlock()
            del self.nextData
        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()
        seqStart, seqEnd = self.packageIndex[self.nextPack]
        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()
        self.r = self.reload_pool.map_async(loadFile, self.seqNames[seqStart:seqEnd])

    def parseNextDataBlock(self):

        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]
        self.phoneLabels = []
        self.snrLabels = []
        self.reverbLabels = []
        speakerSize = 0
        indexSpeaker = 0

        # To accelerate the process a bit
        self.nextData.sort(key=lambda x: (x[0], x[1]))
        tmpData = []

        for speaker, seqName, seq in self.nextData:

            while self.speakers[indexSpeaker] < speaker:
                indexSpeaker += 1
                self.speakerLabel.append(speakerSize)
            if self.speakers[indexSpeaker] != speaker:
                raise ValueError(f"{speaker} invalid speaker")

            if self.phoneLabelsDict is not None:
                self.phoneLabels += self.phoneLabelsDict[seqName]
                newSize = len(self.phoneLabelsDict[seqName]) * self.phoneSize
                seq = seq[:newSize]

                if self.snrDict is not None:
                    self.snrLabels += [self.snrDict[seqName]] * len(
                        self.phoneLabelsDict[seqName]
                    )

                if self.reverbDict is not None:
                    self.reverbLabels += [self.reverbDict[seqName]] * len(
                        self.phoneLabelsDict[seqName]
                    )

            sizeSeq = seq.size(0)
            tmpData.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq
            del seq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(tmpData, dim=0)

    def getPhonem(self, idx):
        idPhone = idx // self.phoneSize
        return self.phoneLabels[idPhone : (idPhone + self.phoneStep)]

    def getSnr(self, idx):
        idPhone = idx // self.snrSize
        return self.snrLabels[idPhone]

    def getReverb(self, idx):
        idPhone = idx // self.reverbSize
        return self.reverbLabels[idPhone]

    def getSpeakerLabel(self, idx):
        idSpeaker = next(x[0] for x in enumerate(self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        outData = self.data[idx : (self.sizeWindow + idx)].view(1, -1)
        label = torch.tensor(self.getSpeakerLabel(idx), dtype=torch.long)
        if self.phoneSize > 0:
            label_phone = torch.tensor(self.getPhonem(idx), dtype=torch.long)
            if not self.doubleLabels:
                label = label_phone
        else:
            label = torch.empty(0)

        if self.snrSize > 0:
            label_snr = torch.tensor(self.getSnr(idx), dtype=torch.float)
        else:
            label_snr = torch.empty(0)

        if self.reverbSize > 0:
            label_reverb = torch.tensor(self.getReverb(idx), dtype=torch.float)
        else:
            label_reverb = torch.empty(0)

        if self.transform is not None:
            outData = self.transform(outData)

        x1, x2 = outData, outData
        if self.augment_past and self.augmentation:
            x1 = self.augmentation(x1)
        if self.augment_future and self.augmentation:
            x2 = self.augmentation(x2)

        x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0)
        outData = torch.cat([x1, x2], dim=0)

        if self.doubleLabels:
            return outData, label, label_phone, label_snr, label_reverb

        return outData, label, label_snr, label_reverb

    def getNSpeakers(self):
        return len(self.speakers)

    def getNSeqs(self):
        return len(self.seqLabel) - 1

    def getNLoadsPerEpoch(self):
        return len(self.packageIndex)

    def getBaseSampler(self, type, batchSize, offset, balance_sampler=None):
        if type == "samespeaker":
            return SameSpeakerSampler(
                batchSize,
                self.speakerLabel,
                self.sizeWindow,
                offset,
                balance_sampler=balance_sampler,
            )
        if type == "samesequence":
            return SameSpeakerSampler(
                batchSize,
                self.seqLabel,
                self.sizeWindow,
                offset,
                balance_sampler=balance_sampler,
            )
        if type == "sequential":
            return SequentialSampler(
                len(self.data), self.sizeWindow, offset, batchSize
            )
        sampler = UniformAudioSampler(len(self.data), self.sizeWindow, offset)
        return BatchSampler(sampler, batchSize, True)

    def getDataLoader(
        self,
        batchSize,
        type,
        randomOffset,
        numWorkers=0,
        onLoop=-1,
        nLoops=-1,
        balance_sampler=None,
    ):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["speaker", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "speaker": grouped sampler speaker-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
        """
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self.loadNextPack()
            nLoops = 1 if nLoops <= 0 else nLoops
        elif nLoops <= 0:
            nLoops = len(self.packageIndex)

        def samplerCall():
            offset = random.randint(0, self.sizeWindow // 2) if randomOffset else 0
            return self.getBaseSampler(
                type, batchSize, offset, balance_sampler
            )

        return AudioLoader(
            self, samplerCall, nLoops, self.loadNextPack, totSize, numWorkers
        )


def loadFile(data):
    speaker, fullPath = data
    seqName = fullPath.stem
    seq = torchaudio.load(fullPath)[0].mean(dim=0)
    return speaker, seqName, seq


class PeakNorm(object):
    def __call__(self, x):
        # Input Size: C x L
        max_val = x.abs().max(dim=1, keepdim=True)[0]
        return x / (max_val + 1e-8)


class AudioLoader(object):
    r"""
    A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """

    def __init__(self, dataset, samplerCall, nLoop, updateCall, size, numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def get_data_loader(self):
        sampler = self.samplerCall()
        return DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=self.numWorkers
        )

    def __iter__(self):

        for i in range(self.nLoop):
            dataloader = self.get_data_loader()

            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):
    def __init__(self, dataSize, sizeWindow, offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class SequentialSampler(Sampler):
    def __init__(self, dataSize, sizeWindow, offset, batchSize):

        self.len = ( dataSize // sizeWindow) // batchSize 
        self.sizeWindow = sizeWindow
        self.offset = offset
        self.startBatches = [x * (dataSize // batchSize) for x in range(batchSize)]
        self.batchSize = batchSize
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        for idx in range(self.len):
            yield [
                self.offset + int(self.sizeWindow * idx) + start
                for start in self.startBatches
            ]

    def __len__(self):
        return self.len


class SameSpeakerSampler(Sampler):
    def __init__(
        self, batchSize, samplingIntervals, sizeWindow, offset, balance_sampler=None
    ):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset
        self.balance_sampler = balance_sampler

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [
            (self.samplingIntervals[i + 1] - self.samplingIntervals[i])
            // self.sizeWindow
            for i in range(nWindows)
        ]

        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]
        self.build_batches()

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow + self.samplingIntervals[iInterval]

    def __iter__(self):
        if self.balance_sampler is not None:
            self.build_batches()
        random.shuffle(self.batches)
        return iter(self.batches)

    def build_batches(self):
        if self.balance_sampler is not None:
            order = self.get_balanced_sampling()
        else:
            order = [
                (x, torch.randperm(val).tolist())
                for x, val in enumerate(self.sizeSamplers)
                if val > 0
            ]

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, len(
                randperm
            )  # self.sizeSamplers[indexSampler]
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                locBatch = [
                    self.getIndex(x, indexSampler)
                    for x in randperm[indexStart:indexEnd]
                ]
                indexStart = indexEnd
                self.batches.append(locBatch)

    def get_balanced_sampling(self):

        target_weights = self.balance_sampler(self.sizeSamplers)
        order = []
        for x, val in enumerate(self.sizeSamplers):
            if val <= 0:
                continue
            to_take = target_weights[
                x
            ]  # int(target_val *self.balance_coeff + (1-self.balance_coeff) * val)
            took = 0
            speaker_batch = []
            while took < to_take:
                remainer = to_take - took
                batch = torch.randperm(val).tolist()
                if remainer < val:
                    batch = batch[:remainer]
                took += len(batch)
                speaker_batch += batch
            order.append((x, speaker_batch))
        return order


def extractLength(couple):
    speaker, locPath = couple
    info = torchaudio.info(str(locPath))[0]
    return info.length


def findAllSeqs(
    dirName, extension=".flac", loadCache=False, speaker_level=1, cache_path=None
):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers

        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index

        outSpeakers
        The speaker labels (in order)

    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension

    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label should be retrieved no matter the
    organization of the dataset.

    """
    if cache_path is None:
        cache_path = str(Path(dirName) / "_seqs_cache.txt")
    if loadCache:
        try:
            outSequences, speakers = torch.load(cache_path)
            print(f"Loaded from cache {cache_path} successfully")
            return outSequences, speakers
        except OSError as err:
            print(f"Ran in an error while loading {cache_path}: {err}")
        print("Could not load cache, rebuilding")

    dirName = str(dirName)
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    for path_file in tqdm.tqdm(Path(dirName).glob(f"**/*{extension}")):

        path_str = str(path_file)
        if not path_str.split("/")[-1].startswith("."):
            speakerStr = (os.sep).join(
                path_str[prefixSize:].split(os.sep)[speaker_level]
            )
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            outSequences.append((speaker, path_str[prefixSize:]))

    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key
    try:
        torch.save((outSequences, outSpeakers), cache_path)
        print(f"Saved cache file at {cache_path}")
    except OSError as err:
        print(f"Ran in an error while saving {cache_path}: {err}")
    return outSequences, outSpeakers


def find_seqs_relative(dirName, extension=".flac"):
    dirName = Path(dirName)
    return [x.relative_to(dirName) for x in dirName.glob(f"**/*{extension}")]


def parseSeqLabels(pathPhone, pathSnr, pathReverb):
    with open(pathPhone, "r") as f:
        lines_phone = f.readlines()

    def replace_label(
        x,
    ):  # converts phoneme labels into boolean (0: silence, 1: speech)
        return np.min([x, 1])

    replace_label_v = np.vectorize(replace_label)

    phone_labels_dict = {"step": 160}  # Step in librispeech dataset is 160bits
    snr_dict = {"step": 160}
    reverb_dict = {"step": 160}

    maxPhone = 0
    for line in lines_phone:
        data = line.split()
        phone_labels_dict[data[0]] = list(replace_label_v([int(x) for x in data[1:]]))
        maxPhone = max(maxPhone, max(phone_labels_dict[data[0]]))

    if pathSnr:
        with open(pathSnr, "r") as f:
            lines_snr = f.readlines()
        for line in lines_snr:
            data = line.split()
            snr_dict[data[0]] = float(data[1])
    else:
        snr_dict = None

    if pathReverb:
        with open(pathReverb, "r") as f:
            lines_reverb = f.readlines()
        for line in lines_reverb:
            data = line.split()
            reverb_dict[data[0]] = float(data[1])
    else:
        reverb_dict = None

    return phone_labels_dict, maxPhone + 1, snr_dict, reverb_dict


def filterSeqs(pathTxt, seqCouples):
    with open(pathTxt, "r") as f:
        inSeqs = [p.replace("\n", "") for p in f.readlines()]

    inSeqs.sort()
    seqCouples.sort(key=lambda x: os.path.splitext(x[1])[0])
    output, index = [], 0
    for x in seqCouples:
        seq = os.path.splitext(x[1])[0]
        while index < len(inSeqs) and seq > inSeqs[index]:
            index += 1
        if index == len(inSeqs):
            break
        if seq == inSeqs[index]:
            output.append(x)
    return output


def glob_relative(path_root: Union[Path, str], search_pattern: str) -> List[Path]:
    return [
        x.relative_to(Path(path_root)) for x in Path(path_root).glob(search_pattern)
    ]
