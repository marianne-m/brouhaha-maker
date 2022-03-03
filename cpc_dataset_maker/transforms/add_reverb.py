import math
import json
import torch
import torchaudio
import random
from pathlib import Path
from torch_audiomentations import ApplyImpulseResponse

from copy import deepcopy
from cpc_dataset_maker.transforms.transform import Transform
from cpc_dataset_maker.transforms.labels import REVERB_LABEL, IMPULSE_RESPONSE_LABEL
from typing import Any, Dict, Set, Tuple, Union
from pathlib import Path


PROBA_NO_REVERB = 0.1
S_TO_MS = 1000


def get_clarity(x, sr, tau):
    n_tau = int(tau * (sr / S_TO_MS))

    nb_frames = x.shape[1]
    if x.shape[0] != 1:
        x = torch.mean(x, dim=0)
    x = x.view(nb_frames)

    # compute C50 from the peak of the impulse response
    t0 = torch.argmax(x).item()
    num = torch.pow(x[t0 : t0 + n_tau], 2).sum().item()
    den = torch.pow(x[t0 + n_tau :], 2).sum().item()
    c_tau = 10 * math.log(num / den, 10)
    return c_tau


class Reverb(Transform):
    def __init__(
        self,
        dir_impulse_response: Union[str, Path],
        tau: float = 50,
        ext_impulse: str = ".wav",
        proba_no_reverb: float = PROBA_NO_REVERB,
    ):
        super().__init__()
        self.proba_no_reverb = proba_no_reverb
        self.dir_impulse_response = Path(dir_impulse_response)
        self.responses = [
            str(self.dir_impulse_response / x)
            for x in self.dir_impulse_response.glob(f"**/*{ext_impulse}")
        ]
        self.tau = tau
        self.sr_ = None
        self.build_c50()

    @property
    def n_responses(self) -> int:
        return len(self.responses)

    @property
    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return {REVERB_LABEL, IMPULSE_RESPONSE_LABEL}

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "proba_no_reverb": self.proba_no_reverb,
            "dir_impulse_response": str(self.dir_impulse_response),
            "responses": self.responses,
            "tau": self.tau,
            "c50list": self.c50list,
        }

    @property
    def sr(self) -> float:
        if self.sr_ is None:
            self.sr_ = torchaudio.info(self.responses[0])[0].rate

        return self.sr_

    def build_c50(self):

        self.c50list = []
        for i in range(self.n_responses):
            audio, sr = torchaudio.load(self.responses[i])
            self.c50list.append(get_clarity(audio, sr, self.tau))

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        new_labels = deepcopy(label_dict)

        if random.random() < self.proba_no_reverb:
            new_labels[REVERB_LABEL] = None
            new_labels[IMPULSE_RESPONSE_LABEL] = None
            return audio_data, new_labels

        index_impulse = random.randint(0, self.n_responses - 1)
        path_impulse = self.responses[index_impulse]
        effect = ApplyImpulseResponse(  # apply reverberation
            ir_paths=[path_impulse], p=1, sample_rate=sr
        )

        new_labels[REVERB_LABEL] = self.c50list[index_impulse]
        new_labels[IMPULSE_RESPONSE_LABEL] = str(self.responses[index_impulse])

        return effect(audio_data.view(1, 1, -1)).view(-1), new_labels
