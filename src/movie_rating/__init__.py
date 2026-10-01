"""电影评分预测课程项目的可复用核心模块。"""

from .baseline import GlobalMeanBaseline, evaluate_predictions
from .comparison import (
    aggregate_ensemble_importance,
    benchmark_model_inference,
    build_metrics_comparison,
    build_unified_predictions,
    calculate_mlp_permutation_importance,
    calculate_regression_metrics,
    calculate_residual_summary,
    calculate_segment_metrics,
    paired_bootstrap_compare,
)
from .data import RawData, load_raw_data, split_ratings, validate_raw_data
from .ensemble import ClippedRegressor, build_model_pipelines
from .features import MovieFeatureEngineer
from .neural import MovieRatingMLP, load_mlp_artifacts, predict_ratings
from .neural_data import MovieRatingDataset, NeuralFeaturePreprocessor, enrich_interactions

__all__ = [
    "GlobalMeanBaseline",
    "ClippedRegressor",
    "MovieFeatureEngineer",
    "MovieRatingMLP",
    "MovieRatingDataset",
    "NeuralFeaturePreprocessor",
    "RawData",
    "aggregate_ensemble_importance",
    "benchmark_model_inference",
    "build_metrics_comparison",
    "build_model_pipelines",
    "build_unified_predictions",
    "calculate_mlp_permutation_importance",
    "calculate_regression_metrics",
    "calculate_residual_summary",
    "calculate_segment_metrics",
    "evaluate_predictions",
    "enrich_interactions",
    "load_raw_data",
    "load_mlp_artifacts",
    "predict_ratings",
    "paired_bootstrap_compare",
    "split_ratings",
    "validate_raw_data",
]
