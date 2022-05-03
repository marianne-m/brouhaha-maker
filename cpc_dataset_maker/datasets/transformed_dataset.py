from pathlib import Path
from typing import Any, Dict, List, Union
from cpc_dataset_maker.datasets.dataset import Dataset
from cpc_dataset_maker.transforms.transform import Transform
from cpc_dataset_maker.transforms.segmentation import Segmentation
from cpc_dataset_maker.transforms.labels import (
    SPEECH_ACTIVITY_LABEL,
    REVERB_LABEL,
    IMPULSE_RESPONSE_LABEL,
    INPUT_PATH_KEY,
    LABELS_KEY,
    OUTPUT_PATH_KEY,
    SNR_LABEL,
    DETAILED_SNR,
)


def update_audio_labels(
    labels_in: List[Dict[str, Any]], new_labels: Dict[str, Any], label_name: str
) -> None:

    for x in labels_in:
        x[LABELS_KEY][label_name] = new_labels[str(x[INPUT_PATH_KEY])]


class TransformDataset(Dataset):
    def __init__(self, root: Union[Path, str], dataset_name: str):
        Dataset.__init__(
            self,
            root=root,
            dataset_name=dataset_name,
        )
        print("Transformed dataset")

    @property
    def path_transform(self) -> Path:
        return self.root / "transform.json"

    def init_audio_labels(self, paths_in: List[Path]) -> List[Dict[str, Any]]:

        self.path_16k.mkdir(exist_ok=True)
        out = []
        for x in paths_in:
            out += [
                {
                    INPUT_PATH_KEY: x,
                    OUTPUT_PATH_KEY: self.path_16k / x.name,
                    LABELS_KEY: {},
                }
            ]

        return out

    def build(
        self,
        labels: List[Dict[str, Any]],
        transform: Union[Segmentation, Transform],
        n_process: int = 10,
        chunksize: int = 10,
    ) -> None:

        print(f"{len(labels)} audio files to transform")
        new_labels = transform.run_on_dataset(
            labels, n_process=n_process, chunksize=chunksize
        )
        transform.save(self.path_transform)

        if SPEECH_ACTIVITY_LABEL in new_labels[0][LABELS_KEY]:
            self.save_voice_activity(
                {
                    x[OUTPUT_PATH_KEY]: x[LABELS_KEY][SPEECH_ACTIVITY_LABEL]
                    for x in new_labels
                }
            )

        if REVERB_LABEL in new_labels[0][LABELS_KEY]:

            self.save_reverb_labels(
                {
                    x[OUTPUT_PATH_KEY]: (
                        x[LABELS_KEY][REVERB_LABEL],
                        x[LABELS_KEY][IMPULSE_RESPONSE_LABEL],
                    )
                    for x in new_labels
                }
            )

        if SNR_LABEL in new_labels[0][LABELS_KEY]:
            self.save_snr_labels(
                {x[OUTPUT_PATH_KEY]: x[LABELS_KEY][SNR_LABEL] for x in new_labels}
            )

        if DETAILED_SNR in new_labels[0][LABELS_KEY]:
            self.save_detailed_snr_labels(
                {x[OUTPUT_PATH_KEY]: x[LABELS_KEY][DETAILED_SNR] for x in new_labels}
            )
