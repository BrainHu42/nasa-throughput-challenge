from collections import defaultdict
from pathlib import Path

VERSION = "v4"

airports = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]

feature_names = [
    "tfm_track",
    "runways",
    "lamp",
    "tbfm",
    "etd",
    "first_position",
    "mfs",
    "configs",
]

# Raw data loaded from file
features_df = defaultdict(dict)

# Configure desired directory structure here
root_directory = Path(".")
data_directory = root_directory / "data"

model_directory = root_directory / "model"
train_directory = data_directory / "train_data"
cache_directory = data_directory / "cache"

prediction_directory = root_directory / "predictions"
submission_format_path = data_directory / "submission_format.csv"
submission_path = root_directory / "submission.csv"


read_csv_kwargs = {
    "configs": {
        "dtype": {
            "departure_runways": str,
            "arrival_runways": str,
        },
        "parse_dates": [
            "datis_time",
            "start_time",
        ],
    },
    "etd": {
        "dtype": {
            "gufi": str,
        },
        "parse_dates": [
            "timestamp",
            "departure_runway_estimated_time",
        ],
    },
    "first_position": {
        "dtype": {"gufi": str},
        "parse_dates": ["time_first_tracked"],
    },
    "lamp": {
        "dtype": {
            "temperature": float,
            "wind_direction": float,
            "wind_speed": float,
            "wind_gust": float,
            "cloud_ceiling": float,
            "visibility": float,
            "cloud": str,
            "lightning_prob": str,
            "precip": bool,
        },
        "parse_dates": [
            "timestamp",
            "forecast_timestamp",
        ],
    },
    "mfs": {
        "dtype": {
            "gufi": str,
            "isarrival": bool,
            "isdeparture": bool,
            "aircraft_engine_class": str,
            "aircraft_type": str,
            "major_carrier": str,
            "flight_type": str,
            "isdeparture": bool,
        },
        "parse_dates": [
            "departure_stand_actual_time",
            "departure_runway_actual_time",
            "arrival_runway_actual_time",
            "arrival_stand_actual_time",
        ],
    },
    "runways": {
        "dtype": {
            "gufi": str,
            "departure_runway_actual": str,
            "arrival_runway_actual": str,
        },
        "parse_dates": [
            "departure_runway_actual_time",
            "arrival_runway_actual_time",
        ],
    },
    "standtimes": {
        "dtype": {
            "gufi": str,
        },
        "parse_dates": [
            "timestamp",
            "arrival_stand_actual_time",
            "departure_stand_actual_time",
        ],
    },
    "tbfm": {
        "dtype": {
            "gufi": str,
        },
        "parse_dates": [
            "timestamp",
            "arrival_runway_sta",
        ],
    },
    "tfm_track": {
        "dtype": {
            "gufi": str,
        },
        "parse_dates": [
            "timestamp",
            "arrival_runway_estimated_time",
        ],
    },
}


from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum


class ContrastiveEnum(Enum):
    simsiam = "simsiam"
    standard = "standard"
    disabled = "disabled"


class DenoisingEnum(Enum):
    standard = "standard"
    disabled = "disabled"


class ProjectionHeadStyleEnum(Enum):
    same = "same"
    different = "different"


class AttentionTypeEnum(Enum):
    col = "col"
    row = "row"
    colrow = "colrow"


@dataclass
class OptimizerConfig:
    """Define the parameters for CutMix augmentation"""

    learning_rate: float = 0.0004  #: value used to specify the learning rate
    other_params: Optional[Dict] = field(default_factory=dict)


@dataclass
class CutMixConfig:
    """Define the parameters for CutMix augmentation"""

    lam: float = 0.1  #: probability original values will be updated


@dataclass
class MixUpConfig:
    """Define the parameters for CutMix augmentation"""

    lam: float = 0.1  #: weight used for the linear combination


@dataclass
class ContrastiveConfig:
    """Define the parameters for Contrastive pretraining task"""

    dropout: float = 0.0  #: probability dropout in projection head
    contrastive_type: ContrastiveEnum = (
        ContrastiveEnum.simsiam
    )  # type of contrastive task to apply
    projhead_style: ProjectionHeadStyleEnum = (
        ProjectionHeadStyleEnum.different
    )  #: it is used to project embeddings
    nce_temp: float = (
        0.5  #: temperature used for the logits in case of standard constrastive type
    )
    weight: float = 0.1  #: weight of the loss for this pretraining task


@dataclass
class DenoisingConfig:
    """Define the parameters for Denoising pretraining task"""

    denoising_type: DenoisingEnum = (
        DenoisingEnum.standard
    )  # type of denoising task to apply
    weight_cross_entropy: float = (
        0.5  #: weight reconstruction loss for categorical features
    )
    weight_mse: float = 0.5  #: weight reconstruction loss for continuous features
    scale_dim_internal_sepmlp: float = (
        5  # scale factor of the input dimension for the first linear layer
    )
    dropout: float = 0.0  #: probability dropout in SepMLP


@dataclass
class AugmentationConfig:
    """Define the parameters used for the augmentations"""

    cutmix: Optional[CutMixConfig] = CutMixConfig()
    mixup: Optional[MixUpConfig] = MixUpConfig()


@dataclass
class PreTrainTaskConfig:
    """Define the parameters used for pretraining tasks"""

    contrastive: Optional[ContrastiveConfig] = ContrastiveConfig()
    denoising: Optional[DenoisingConfig] = DenoisingConfig()


@dataclass
class PreTrainConfig:
    """Define parameters for the steps used during the pretraining"""

    aug: AugmentationConfig = AugmentationConfig()
    task: PreTrainTaskConfig = PreTrainTaskConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    epochs: int = 2  #: number of epochs of training phase
    batch_size: int = 256  #: dimension of batches using by dataloaders


@dataclass
class TrainConfig:
    """Define parameters for the steps used during the training"""

    internal_dimension_output_layer: int = (
        20  #: internal dimension of the MLP that compute the output
    )
    mlpfory_dropout: float = (
        0.1  #: probability dropout in the the MLP used for prediction
    )
    epochs: int = 40  #: number of epochs of training phase
    optimizer: OptimizerConfig = OptimizerConfig()
    batch_size: int = 256  #: dimension of batches using by dataloaders


@dataclass
class TransformerConfig:
    depth: int = 4  #: number of attention blocks used in the transformer
    heads: int = 2  #: number of heads used in the transformer
    dropout: float = 0.1  #: probability dropout in the transformer
    attention_type: AttentionTypeEnum = AttentionTypeEnum.col  #: type of attention
    dim_head: int = 64
    scale_dim_internal_col: float = (
        4  # scale factor of the input dimension in case of attention_type col
    )
    scale_dim_internal_row: float = (
        4  # scale factor of the input dimension in case of attention_type row
    )


@dataclass
class NetworkConfig:
    """Define the neural network parameters"""

    transformer: TransformerConfig = TransformerConfig()
    num_workers: int = 8  #: number of cores to use
    embedding_size: int = 10  #: dimension of computed embeddings
    internal_dimension_embed_continuous: int = (
        100  #: internal dimension of the mlp used to project continuous columns
    )
    dropout_embed_continuous: float = (
        0.0  #: dropout used to compute embedding continuous features
    )


@dataclass
class SaintConfig:
    """Define all the parameters used in SAINT"""

    network: NetworkConfig = NetworkConfig()
    pretrain: PreTrainConfig = PreTrainConfig()
    train: TrainConfig = TrainConfig()


train_params = {
    "KATL": {
        "iterations": 9100,
        "depth": 9,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "var_arrival_error",
            "tbfm_arrivals_diff",
            "avg_estimate_change",
            "max_precip_3h",
            "day_cos",
            "day_sin",
            "num_departure_runways",
            "dow_cos",
            "tbfm_avg_arrival_error",
            "max_visibility_3h",
            "var_time_between_arrivals",
            "avg_arrival_error",
            "avg_departure_taxi_time",
            "max_wind_gust_3h",
            "tbfm_avg_estimate_change",
            "max_cloud_3h",
            "max_temperature_3h",
            "tfm_arrivals_next",
            "etd_departures",
            "num_arrival_runways",
        ],
    },
    "KCLT": {
        "iterations": 10500,
        "depth": 9,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "num_arrival_runways",
            "max_temperature_3h",
            "avg_departure_taxi_time",
            "tbfm_arrivals_diff",
            "day_sin",
            "max_lightning_prob_3h",
            "num_departure_runways",
            "tbfm_tfm_time_diff",
            "day_cos",
            "max_cloud_ceiling_3h",
            "var_time_between_arrivals",
            "tbfm_var_arrival_error",
            "tbfm_var_time_between_estimated_arrivals",
            "dow_cos",
            "tbfm_avg_time_between_estimated_arrivals",
            "etd_departures",
        ],
    },
    "KDEN": {
        "iterations": 6400,
        "depth": 9,
        "l2_leaf_reg": 5,
        "dropped_features": [
            "num_arrival_runways",
            "max_cloud_ceiling_3h",
            "tbfm_tfm_time_diff",
            "max_temperature_3h",
            "var_time_between_arrivals",
            "dow_sin",
            "avg_arrival_runway_throughput",
            "day_cos",
            "max_wind_gust_3h",
            "avg_departure_taxi_time",
            "day_sin",
            "avg_time_between_arrivals",
            "tbfm_var_arrival_error",
            "min_sin",
            "var_time_between_estimated_arrivals",
            "tbfm_arrivals_diff_2",
            "avg_arrival_taxi_time",
            "arrivals_mode",
            "tbfm_var_time_between_estimated_arrivals",
            "num_departure_runways",
            "etd_departures",
            "dow_cos",
            "month_sin",
        ],
    },
    "KDFW": {
        "iterations": 9500,
        "depth": 7,
        "l2_leaf_reg": 7,
        "dropped_features": [
            "avg_time_between_arrivals",
            "dow_sin",
            "max_cloud_ceiling_3h",
            "arrivals_mean_abs_diff",
            "dow_cos",
            "num_departure_runways",
            "avg_arrival_runway_throughput",
            "max_wind_gust_3h",
            "max_precip_3h",
            "max_cloud_3h",
            "day_cos",
            "var_time_between_arrivals",
            "max_temperature_3h",
        ],
    },
    "KJFK": {
        "iterations": 10200,
        "depth": 7,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "tbfm_arrivals_diff_2",
            "arrivals_mode",
            "etd_departures",
            "avg_arrival_runway_throughput",
            "avg_time_between_estimated_arrivals",
            "dow_cos",
            "arrivals_diff",
            "max_cloud_3h",
            "avg_departure_taxi_time",
            "tbfm_var_arrival_error",
            "avg_time_to_arrival",
            "day_sin",
            "min_sin",
            "num_arrival_runways",
            "max_cloud_ceiling_3h",
            "day_cos",
        ],
    },
    "KMEM": {
        "iterations": 10000,
        "depth": 9,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "num_departure_runways",
            "tbfm_arrivals_diff_2",
            "tbfm_var_arrival_error",
            "arrivals_mean_abs_diff",
            "day_sin",
            "max_lightning_prob_3h",
            "var_time_between_arrivals",
            "avg_departure_taxi_time",
            "avg_arrival_taxi_time",
            "avg_time_between_arrivals",
            "tbfm_var_time_between_estimated_arrivals",
            "max_wind_gust_3h",
            "arrivals_diff",
            "etd_departures",
            "avg_arrival_runway_throughput",
            "max_cloud_ceiling_3h",
            "tbfm_arrivals_diff",
            "min_sin",
            "tfm_arrivals_diff_2",
            "max_cloud_3h",
        ],
    },
    "KMIA": {
        "iterations": 7700,
        "depth": 9,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "arrivals_mean_abs_diff",
            "tbfm_arrivals_diff_2",
            "arrivals_mode",
            "avg_arrival_runway_throughput",
            "tbfm_tfm_time_diff",
            "dow_cos",
            "max_temperature_3h",
            "tbfm_var_time_between_estimated_arrivals",
            "day_sin",
            "var_time_between_arrivals",
            "day_cos",
            "tbfm_var_arrival_error",
            "etd_departures",
            "num_arrival_runways",
            "avg_time_to_arrival",
        ],
    },
    "KORD": {
        "iterations": 13200,
        "depth": 8,
        "l2_leaf_reg": 10,
        "dropped_features": [
            "arrivals_mean_abs_diff",
            "max_visibility_3h",
            "tbfm_tfm_time_diff",
            "day_sin",
            "num_departure_runways",
            "dow_sin",
            "tbfm_var_arrival_error",
            "tbfm_var_time_between_estimated_arrivals",
            "max_cloud_ceiling_3h",
            "avg_arrival_taxi_time",
            "var_arrival_error",
            "max_cloud_3h",
        ],
    },
    "KPHX": {
        "iterations": 6100,
        "depth": 8,
        "l2_leaf_reg": 4,
        "dropped_features": [
            "tfm_tbfm_diff",
            "max_cloud_ceiling_3h",
            "tbfm_var_arrival_error",
            "tfm_arrivals_diff_2",
            "num_arrival_runways",
            "num_departure_runways",
            "tbfm_tfm_time_diff",
            "avg_flight_time",
            "day_sin",
            "var_time_between_arrivals",
            "tbfm_arrivals_diff_2",
            "arrivals_mean_abs_diff",
            "day_cos",
        ],
    },
    "KSEA": {
        "iterations": 6700,
        "depth": 8,
        "l2_leaf_reg": 5,
        "dropped_features": [
            "tbfm_tfm_time_diff",
            "dow_sin",
            "avg_departure_taxi_time",
            "day_sin",
            "max_wind_gust_3h",
            "day_cos",
            "dow_cos",
            "max_lightning_prob_3h",
            "max_cloud_3h",
            "num_departure_runways",
            "tbfm_avg_time_between_estimated_arrivals",
            "etd_departures",
            "max_precip_3h",
            "num_arrival_runways",
        ],
    },
}
