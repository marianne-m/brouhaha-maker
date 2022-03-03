from cpc_dataset_maker.datasets.allstar import ALLSTAR
from cpc_dataset_maker.datasets.buckeye import Buckeye
from cpc_dataset_maker.datasets.coraal import CORAAL
from cpc_dataset_maker.datasets.dataset import Dataset
from cpc_dataset_maker.datasets.librispeech import LibriSpeechCPC

AVAILABLE_DATASETS = ["allsstar", "buckeye", "coraal", "librispeech_cpc", "standard"]


def get_dataset_builder(dataset_name: str) -> Dataset:

    if dataset_name == "standard":
        return Dataset
    if dataset_name == "allsstar":
        return ALLSTAR
    if dataset_name == "buckeye":
        return Buckeye
    if dataset_name == "coraal":
        return CORAAL
    if dataset_name == "librispeech_cpc":
        return LibriSpeechCPC

    raise ValueError(f"Invalid dataset name {dataset_name}")
