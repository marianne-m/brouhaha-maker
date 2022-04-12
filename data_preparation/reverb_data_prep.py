"""
This script downloads the MIT Acoustical Reverberation Scene Statistics Survey and
Echo Thief Impulse Response Library that are used as our reverbaration dataset.
The dataset is then split into a train set and a dev set with a 80/20 ratio by default.
By default, the dataset is downloaded in this scipt's directory.

Usage : python reverb_data_prep.py
Usage (with options) :
    python reverb_data_prep.py --dataset-path path/to/desired/dataset --dev-percentage desired_percentage

"""
import sys
import os
import argparse
import shutil
import glob
import zipfile
import random

import wget


def parse_args(argv):
    """Parser"""
    parser = argparse.ArgumentParser(description='Download an prepare reverberation datasets')

    parser.add_argument('--dataset-path', type=str, default=None,
                        help="Path where the dataset will be download")
    parser.add_argument('--dev-percentage', type=int, default="20",
                        help="Percentage of files in the dev set. Default : 20.")

    return parser.parse_args(argv)


def main(argv):
    """
    Download and unzip the dataset, create train set and
    dev set.
    """
    args = parse_args(argv)

    if args.dataset_path:
        os.makedirs(args.dataset_path, exist_ok=True)
        os.chdir(args.dataset_path)

    print("Downloading MIT Acoustical Reverberation Scene Statistics Survey and " \
          "Echo Thief Impulse Response Library...")
    for url in ['http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip', \
        "http://www.echothief.com/wp-content/uploads/2016/06/EchoThiefImpulseResponseLibrary.zip"]:
        wget.download(url)

    print("\nUnzipping zipfiles...")
    with zipfile.ZipFile("Audio.zip","r") as zip_ref:
        zip_ref.extractall(".")

    with zipfile.ZipFile("EchoThiefImpulseResponseLibrary.zip","r") as zip_ref:
        zip_ref.extractall(".")

    print("Done unzipping")

    echothief_files = glob.glob("EchoThiefImpulseResponseLibrary/**/*.wav")
    for file in echothief_files:
        shutil.move(file, 'Audio')

    random.seed(2)

    print("Creating train set and dev set...")
    os.makedirs('dev', exist_ok=True)

    # Creating the dev set by moving the files
    impulse_responses = glob.glob('Audio/*.wav')
    number_of_dev = int(args.dev_percentage*len(impulse_responses)/100)
    dev_set = random.sample(impulse_responses, number_of_dev)
    for file in dev_set:
        shutil.move(file, 'dev')

    # Only train files remain in the Audio folder
    os.rename('Audio', 'train')

    print("Done.")

    # Clean up
    os.remove("Audio.zip")
    os.remove("EchoThiefImpulseResponseLibrary.zip")
    shutil.rmtree("__MACOSX")
    shutil.rmtree("EchoThiefImpulseResponseLibrary")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
