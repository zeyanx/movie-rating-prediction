"""网页配置读取与约束验证。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "stage6_app.json"


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "app_title", "default_user_id", "default_model", "fallback_model",
        "default_top_n", "max_top_n", "minimum_popularity_count",
        "popularity_prior_count", "recommendation_weights",
        "reference_timestamp_strategy", "database_path",
        "model_recommendation_path", "model_lab_models",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"第6阶段配置缺少字段：{missing}")
    weights = config["recommendation_weights"]
    expected_weights = {"mlp_prediction", "genre_preference", "bayesian_popularity"}
    if set(weights) != expected_weights:
        raise ValueError("recommendation_weights 字段不完整")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("推荐权重不能为负数")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("推荐权重之和必须为1")
    if not 1 <= int(config["default_user_id"]) <= 943:
        raise ValueError("default_user_id 必须位于1至943")
    if not 1 <= int(config["default_top_n"]) <= int(config["max_top_n"]) <= 20:
        raise ValueError("推荐数量配置必须满足 1 <= default <= max <= 20")
    if int(config["minimum_popularity_count"]) < 0:
        raise ValueError("minimum_popularity_count 不能为负数")
    if int(config["popularity_prior_count"]) <= 0:
        raise ValueError("popularity_prior_count 必须为正数")
    if config["reference_timestamp_strategy"] != "user_latest_or_dataset_max":
        raise ValueError("网页预测必须使用训练期历史时间戳策略")


@lru_cache(maxsize=4)
def load_app_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取并验证网页配置；仅缓存不可变的配置内容。"""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到网页配置：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    _validate_config(config)
    return config


def resolve_project_path(relative_path: str | Path) -> Path:
    """将配置中的相对路径安全解析到项目根目录。"""
    candidate = (PROJECT_ROOT / Path(relative_path)).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"路径必须位于项目目录内：{relative_path}") from exc
    return candidate
