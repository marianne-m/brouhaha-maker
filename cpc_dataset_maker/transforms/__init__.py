from numpy import kaiser
from cpc_dataset_maker.transforms.extend_silences import ExtendSilenceTransform
from cpc_dataset_maker.transforms.add_reverb import Reverb, PROBA_NO_REVERB
from cpc_dataset_maker.transforms.add_noise import AddNoise
from cpc_dataset_maker.transforms.normalization import PeakNorm, EnergyNorm

AVAILABLE_TRANSFORMS = ["extend_sil", "reverb", "peaknorm", "energynorm", "noise"]


def get_transform(transform_type: str, **kwargs):

    if transform_type == "extend_sil":
        return ExtendSilenceTransform(
            cossfade_sec=kwargs["cossfade_sec"],
            sil_mean_sec=kwargs["sil_mean_sec"],
            target_share_sil=kwargs["target_share_sil"],
            expand_silence_only=kwargs["expand_silence_only"],
            sil_min_sec=kwargs["silence-min-duration"]
        )
    if transform_type == "reverb":
        return Reverb(
            dir_impulse_response=kwargs["dir_impulse_responses"],
            tau=kwargs["tau"],
            ext_impulse=kwargs["ext_impulse"],
            proba_no_reverb=kwargs["proba_no_reverb"],
        )
    if transform_type == "noise":
        return AddNoise(
            dir_noise=kwargs["dir_noise"],
            ext_noise=kwargs["ext_noise"],
            snr_min=kwargs["snr_min"],
            snr_max=kwargs["snr_max"],
        )
    if transform_type == "peaknorm":
        return PeakNorm()
    if transform_type == "energynorm":
        return EnergyNorm()

    raise ValueError(f"Invalid transformation name : {transform_type}")
