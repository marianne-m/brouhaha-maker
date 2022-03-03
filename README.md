# Building the datasets for the VAD / SNR predictors

Here are the instruction to build the extended datasets for both training and testing the VAD / SNR prediction model.

## Downloading the datasets

### LibriSpeech 1000

LibriSpeech 1000, composed of multiple dataset of librispeech, is used.

Start by downloading train-clean-100, train-clean-360 and train-other-500 on [this page](https://www.openslr.org/12/).
Then, you will need to download [Van Den Oord](https://arxiv.org/abs/1807.03748) labels [here]((https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb).

Your Librispeech directory should have the following structure:

```
Librispeech
    |
    train-clean-100
        |
        speaker_1
            |
            chapter_1
                |
                audio_speaker1_chapter1.wav
                |...
            ...
        ...
    |
    LibriSpeech100_labels_split
        |
        converted_aligned_phones.txt
        test_split.txt
        train_split.txt
```


## Building the data

To build the data, you need to run the script: `build_vad_datasets.py`

```
python build_vad_datasets.py init $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             --root-in $DIR_DOWNLOADED_DATA
```

# Launching pyannote on a dataset

## Inference

To run a pyannote inference on a dataset, you can use the script `vad_pyannote/launch_vad_pyannote.py`:

```
python vad_pyannote/launch_vad_pyannote.py ${DATASET_DIR}/audio_16k \
                                            --file_extension .flac
                                            -o ${DATASET_DIR}/vad_pyannote
```

This script takes advantage of all available GPUs. You can launch it on scrum to deal efficiently with large dataset.

## Training

TODO

# Apply diverse transformations the dataset

To transform your dataset, you will need to use `build_vad_datasets.py` as follow:

```
python build_vad_datasets.py transform $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             -o $OUTPUT_DIR_TRANSFORM \
                            --transforms [ TRANSFORM_COMBINATON ]
```

## Noise Augmentation

You will need to audioset [AUDIOSET](https://research.google.com/audioset/dataset/index.html). To launch the noise augmentation use `build_vad_datasets.py` as follow:

```
python build_vad_datasets.py transform $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             --name noise \
                             --transforms noise \
                             --dir-noise $MUSAN_DIR \
                             --ext-noise .wav \
                             -o $OUTPUT_DIR_TRANSFORM \
```

## Reverb augmentation

First, you will need to download impulse datasets to perform a convincing reverberation. We used the [MIT Acoustical Reverberation Scene Statistics Survey](http://mcdermottlab.mit.edu/Reverb/IR_Survey.html) for the train set and [EchoThief](http://www.echothief.com/downloads/) for the train set.

CPC works with 16kHz audio files, but these reverb datasets have an higher sample rate. To convert them to 16kHz run `build_vad_datasets` again :

```
python build_vad_datasets.py init standard \
                             $OUTPUT_DIR_IR_DATASET \
                             --root-in $DIR_DOWNLOADED_DATA
```

Then, to apply the reverberation:

```
python build_vad_datasets.py transform $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             --name reverb
                             -o $OUTPUT_DIR_TRANSFORM \
                            --transforms reverb \
                            --dir-impulse-response $OUTPUT_DIR_IR_DATASET
```

## Combining several transformation

You can combine different transformations any way you want. For example, to run a peak normalization, followed by some reverb augmentation and finish with noise augmentation run:

```
python build_vad_datasets.py transform $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             --name combo \
                             -o $OUTPUT_DIR_TRANSFORM \
                            --transforms peaknorm reverb noise \
                            --dir-impulse-response $OUTPUT_DIR_IR_DATASET \
                            --dir-noise $MUSAN_DIR \
                            --ext-noise .wav 
```

# Segment a dataset into smaller segments

You can segment a dataset into smaller audio segments using `build_vad_datasets.py`:

```
python build_vad_datasets.py segment $DATASET_NAME \
                             $OUTPUT_DIR_VAD_DATASET \
                             -o $OUTPUT_DIR_SEGMENT \
                            -t target_size_segment
```
# Additional ressources:

Google drive: https://drive.google.com/drive/folders/1XXc8526sIsfg6w8h7oOUF9fWC-9ap2Uu?usp=sharing

# What's next ?

- [ ] Fix noise augmentation :
    - either AddNoise:
        - Add self.max_size_loaded (which remains constant and indicates the amount of noise data that is being loaded at once)
        - Add self.cumulated_duration (which is updated after each run of the __call__ function and describes the cumulated duration of segments that have been corrupted with additive noise)
        - Once self.cumulated_duration reaches self.max_size_loaded, call to self.load_noise_db() that must load M segments of noise until self.max_sized_loaded is reached
    - Or use Marvin's technics and pre-process noise, concatenate in four big files with cross fading, and use these.


- [ ] AddNoise should call AddReverb to corrupt noise segments with reverberation
- [ ] No need to apply VAD : Audioset already has the labels
