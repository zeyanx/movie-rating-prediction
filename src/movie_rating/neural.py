"""PyTorch电影评分MLP及可供后续Streamlit复用的推理接口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .neural_data import MovieRatingDataset, NeuralFeaturePreprocessor


class MovieRatingMLP(nn.Module):
    """融合用户/电影Embedding、静态属性、类型与时间特征的MLP。"""

    def __init__(
        self,
        vocabulary_sizes: dict[str, int],
        numeric_feature_count: int,
        genre_feature_count: int,
        user_embedding_dim: int = 32,
        movie_embedding_dim: int = 48,
        occupation_embedding_dim: int = 8,
        gender_embedding_dim: int = 2,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
        rating_global_mean: float = 3.5,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims不能为空")
        self.model_config = {
            "vocabulary_sizes": {key: int(value) for key, value in vocabulary_sizes.items()},
            "numeric_feature_count": int(numeric_feature_count),
            "genre_feature_count": int(genre_feature_count),
            "user_embedding_dim": int(user_embedding_dim),
            "movie_embedding_dim": int(movie_embedding_dim),
            "occupation_embedding_dim": int(occupation_embedding_dim),
            "gender_embedding_dim": int(gender_embedding_dim),
            "hidden_dims": [int(value) for value in hidden_dims],
            "dropout": float(dropout),
            "rating_global_mean": float(rating_global_mean),
        }

        self.user_embedding = nn.Embedding(vocabulary_sizes["user"], user_embedding_dim)
        self.movie_embedding = nn.Embedding(vocabulary_sizes["movie"], movie_embedding_dim)
        self.occupation_embedding = nn.Embedding(
            vocabulary_sizes["occupation"], occupation_embedding_dim
        )
        self.gender_embedding = nn.Embedding(vocabulary_sizes["gender"], gender_embedding_dim)
        self.user_bias = nn.Embedding(vocabulary_sizes["user"], 1)
        self.movie_bias = nn.Embedding(vocabulary_sizes["movie"], 1)

        input_dim = (
            user_embedding_dim + movie_embedding_dim + occupation_embedding_dim
            + gender_embedding_dim + numeric_feature_count + genre_feature_count
        )
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.ReLU(), nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights(rating_global_mean)

    def _initialize_weights(self, rating_global_mean: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        ratio = np.clip((float(rating_global_mean) - 1.0) / 4.0, 1e-4, 1.0 - 1e-4)
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, float(np.log(ratio / (1.0 - ratio))))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        combined = torch.cat([
            self.user_embedding(batch["user_index"]),
            self.movie_embedding(batch["movie_index"]),
            self.occupation_embedding(batch["occupation_index"]),
            self.gender_embedding(batch["gender_index"]),
            batch["numeric"], batch["genres"],
        ], dim=1)
        raw = self.network(combined).squeeze(1)
        raw = raw + self.user_bias(batch["user_index"]).squeeze(1)
        raw = raw + self.movie_bias(batch["movie_index"]).squeeze(1)
        return 1.0 + 4.0 * torch.sigmoid(raw)


def build_model_from_metadata(metadata: dict[str, Any]) -> MovieRatingMLP:
    """根据JSON元数据重建模型结构。"""
    return MovieRatingMLP(**metadata["model_config"])


def save_model_state(model: MovieRatingMLP, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_mlp_artifacts(
    model_path: Path | str,
    preprocessor_path: Path | str,
    metadata_path: Path | str,
    device: str = "cpu",
) -> tuple[MovieRatingMLP, NeuralFeaturePreprocessor, dict[str, Any]]:
    """加载最终权重、预处理器和结构元数据，默认在CPU推理。"""
    target_device = torch.device(device)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    preprocessor = NeuralFeaturePreprocessor.load(Path(preprocessor_path))
    model = build_model_from_metadata(metadata)
    state = torch.load(Path(model_path), map_location=target_device, weights_only=True)
    model.load_state_dict(state)
    model.to(target_device)
    model.eval()
    return model, preprocessor, metadata


def predict_ratings(
    model: MovieRatingMLP,
    preprocessor: NeuralFeaturePreprocessor,
    frame: pd.DataFrame,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """对单条或批量合并后交互进行评分预测。"""
    if len(frame) == 0:
        return np.asarray([], dtype=np.float32)
    dataset = MovieRatingDataset(preprocessor.transform(frame))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: tensor.to(target_device) for name, tensor in batch.items()}
            outputs.append(model(batch).detach().cpu().numpy())
    return np.clip(np.concatenate(outputs).astype(np.float32), 1.0, 5.0)
