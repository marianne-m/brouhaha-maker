from copy import deepcopy
import json
from pathlib import Path
from functools import partial
import torch
import torchaudio
import tqdm
from torch.multiprocessing import Pool
from typing import Any, Dict, List, Set, Tuple, Union
from cpc_dataset_maker.transforms.labels import (
    INPUT_PATH_KEY,
    LABELS_KEY,
    OUTPUT_PATH_KEY,
)


class Transform:
    def __init__(self) -> None:
        pass

    @property
    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return set()

    @property
    def json_params(self) -> Dict[str, Any]:
        return {"name": self.__class__.__name__, "init-params": self.init_params}

    @property
    def init_params(self) -> Dict[str, Any]:
        raise RuntimeError(
            f"Init parameters not implemented for {self.__class__.__name__}"
        )

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any], detailed_path: Path
    ) -> Tuple[torch.tensor, Dict[str, Any]]:
        raise RuntimeError(
            f"Call function not implemented for {self.__class__.__name__}"
        )

    def save(self, path_out: Union[str, Path]) -> None:

        with open(path_out, "w") as f_:
            json.dump(self.json_params, f_, indent=2)

    def _run_on_file(self, data: Dict[str, Any], path_detailed: Path) -> Dict[str, Any]:

        path_in = data[INPUT_PATH_KEY]
        path_out = data[OUTPUT_PATH_KEY]
        labels = data[LABELS_KEY]

        path_out.parent.mkdir(exist_ok=True)
        detailed_out = Path(path_detailed / path_in.stem)
        detailed_out.mkdir(exist_ok=True)

        audio, sr = torchaudio.load(str(path_in))
        audio = audio.mean(dim=0)
        torchaudio.save(detailed_out / "original_speech.flac", audio.view(1, -1), sr)
        new_audio, new_labels = self.__call__(audio, sr, labels, detailed_out)
        torchaudio.save(str(path_out), new_audio.view(1, -1), sr)
        return {OUTPUT_PATH_KEY: str(path_out), LABELS_KEY: new_labels}

    def run_on_dataset(
        self,
        list_path_labels: List[Dict[str, Any]],
        n_process: int = 10,
        chunksize: int = 10,
        path_detailed: Path = None
    ) -> List[Dict[str, Any]]:

        with Pool(n_process) as p:
            r = list(
                tqdm.tqdm(
                    p.imap(partial(self._run_on_file, path_detailed=path_detailed), list_path_labels, chunksize=chunksize),
                    total=len(list_path_labels),
                )
            )

        return r


class CombinedTransform(Transform):
    def __init__(self, list_transform: List[Transform]):
        super(CombinedTransform, self).__init__()

        self.transforms = deepcopy(list_transform)

    @property
    def input_labels(self) -> Set[str]:
        base = set()
        for t in self.transforms:
            base = base.union(t.input_labels)
        return base

    @property
    def output_labels(self) -> Set[str]:
        base = set()
        for t in self.transforms:
            base = base.union(t.output_labels)
        return base

    @property
    def init_params(self) -> Dict[str, Any]:
        return [x.json_params for x in self.transforms]

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any], detailed_path: Path
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        new_audio = audio_data
        new_labels = label_dict
        for transform in self.transforms:
            new_audio, new_labels = transform(new_audio, sr, new_labels, detailed_path)

        return new_audio, new_labels
