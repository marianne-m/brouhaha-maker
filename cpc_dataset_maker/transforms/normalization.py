import torch
from typing import Any, Dict, Set, Tuple
from cpc_dataset_maker.transforms.transform import Transform


def energy_normalization(wav, epsilon: float = 1e-8):
    return wav / (torch.sqrt(torch.mean(wav ** 2)) + epsilon)


def peak_normalization(wav, epsilon: float = 1e-8):
    return wav / (wav.abs().max(dim=0, keepdim=True)[0] + epsilon)


class PeakNorm(Transform):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    property

    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return set()

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
        }

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        new_audio = peak_normalization(audio_data, self.epsilon)
        return new_audio, label_dict


class EnergyNorm(Transform):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    property

    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return set()

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
        }

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        new_audio = energy_normalization(audio_data, self.epsilon)
        return new_audio, label_dict
