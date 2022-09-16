from cpc_dataset_maker.datasets.dataset import Dataset
from cpc_dataset_maker.datasets.librispeech import LibriSpeechCPC
from cpc_dataset_maker.datasets.coraal import CORAAL

AVAILABLE_DATASETS = ["librispeech_cpc", "standard"]


def get_dataset_builder(dataset_name: str) -> Dataset:

    if dataset_name == "standard":
        return Dataset
    if dataset_name == "librispeech_cpc":
        return LibriSpeechCPC
    if dataset_name =="coraal":
        return CORAAL

    raise ValueError(f"Invalid dataset name {dataset_name}")
