from cpc_dataset_maker.datasets.dataset import Dataset
from typing import Optional, Union
from pathlib import Path


class LibriSpeechCPC(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
    ):
        Dataset.__init__(
            self,
            root=root,
            dataset_name="librispeech_cpc",
        )
        print(f"Working with LibriSpeechCPC")

    def create_phone_labels(self, path_phone_labels: Union[Path, str]) -> None:
        self.use_phone_labels(path_phone_labels)

    def create_rttm(self, path_phone_labels: Union[Path, str]) -> None:
        self.use_phone_labels(path_phone_labels)
        self.create_rttm_from_phone_labels()

    def build_from_root_dir(self, root_dir: Union[Path, str]) -> None:
        path_phone_labels = (
            root_dir / "LibriSpeech100_labels_split" / "converted_aligned_phones.txt"
        )
        path_train = root_dir / "LibriSpeech100_labels_split" / "train_split.txt"
        path_test = root_dir / "LibriSpeech100_labels_split" / "test_split.txt"

        self.use_training_set(path_train)
        self.use_test_set(path_test)
        self.create_rttm(path_phone_labels)
        self.resample(root_dir / "train-clean-100")
