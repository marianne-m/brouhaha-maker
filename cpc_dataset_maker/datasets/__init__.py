from cpc_dataset_maker.datasets.dataset import Dataset
from cpc_dataset_maker.datasets.librispeech import LibriSpeechCPC

AVAILABLE_DATASETS = ["librispeech_cpc", "standard"]


def get_dataset_builder(dataset_name: str) -> Dataset:

    if dataset_name == "standard":
        return Dataset
    if dataset_name == "librispeech_cpc":
        return LibriSpeechCPC

    raise ValueError(f"Invalid dataset name {dataset_name}")
