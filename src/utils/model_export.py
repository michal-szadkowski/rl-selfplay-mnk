import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import torch

from alg.architectures.cnn import CnnSActorCritic, CnnLActorCritic
from alg.architectures.resnet import ResNetSActorCritic, ResNetLActorCritic
from alg.architectures.transformer import (
    TransformerSActorCritic,
    TransformerLActorCritic,
)
from alg.architectures.configs import (
    CnnSActorCritic as CnnBSActorCritic,
    CnnLActorCritic as CnnBLActorCritic,
    ResNetSActorCritic as ResNetBSActorCritic,
    ResNetLActorCritic as ResNetBLActorCritic,
    TransformerSActorCritic as TransformerBSActorCritic,
    TransformerLActorCritic as TransformerBLActorCritic,
)
from alg.architectures.sgrtransformer import (
    TransformerSActorCritic as TransformerCSActorCritic,
    TransformerLActorCritic as TransformerCLActorCritic,
)


ARCHITECTURE_REGISTRY: Dict[str, Callable[..., torch.nn.Module]] = {
    "cnn_s": CnnSActorCritic,
    "cnn_l": CnnLActorCritic,
    "resnet_s": ResNetSActorCritic,
    "resnet_l": ResNetLActorCritic,
    "transformer_s": TransformerSActorCritic,
    "transformer_l": TransformerLActorCritic,
    "cnn_b_s": CnnBSActorCritic,
    "cnn_b_l": CnnBLActorCritic,
    "resnet_b_s": ResNetBSActorCritic,
    "resnet_b_l": ResNetBLActorCritic,
    "transformer_b_s": TransformerBSActorCritic,
    "transformer_b_l": TransformerBLActorCritic,
    "transformer_c_s": TransformerCSActorCritic,
    "transformer_c_l": TransformerCLActorCritic,
}


@dataclass
class ModelMetadata:
    """Metadata stored alongside exported models."""

    model_id: str
    iteration: int
    architecture_name: str
    architecture_params: Dict[str, Any]
    export_timestamp: str
    is_benchmark_breaker: bool
    run_name: Optional[str]
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "iteration": self.iteration,
            "architecture": {
                "name": self.architecture_name,
                "params": self.architecture_params,
            },
            "export_timestamp": self.export_timestamp,
            "is_benchmark_breaker": self.is_benchmark_breaker,
            "run_name": self.run_name,
        }
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        architecture = data.get("architecture", {})
        return cls(
            model_id=data["model_id"],
            iteration=data.get("iteration", 0),
            architecture_name=architecture.get("name"),
            architecture_params=architecture.get("params", {}),
            export_timestamp=data.get("export_timestamp", ""),
            is_benchmark_breaker=data.get("is_benchmark_breaker", False),
            run_name=data.get("run_name"),
        )


class ModelExporter:
    def __init__(self, run_name: str = None, base_dir: str = "models"):
        self.run_name = run_name or self._generate_timestamp()
        self.export_dir = os.path.join(base_dir, self.run_name)
        os.makedirs(self.export_dir, exist_ok=True)

    def _generate_timestamp(self) -> str:
        """Generate timestamp in format YYYYMMDD_HHMMSS."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_model(
        self,
        network: torch.nn.Module,
        iteration: int,
        is_benchmark_breaker: bool = False,
    ) -> str:
        """Export a model with automatic architecture detection."""
        self._validate_exportable(network)

        model_id = f"model_{iteration:05d}"
        model_path = os.path.join(self.export_dir, f"{model_id}.pt")
        metadata_path = os.path.join(self.export_dir, f"{model_id}.json")

        torch.save(network.state_dict(), model_path)

        metadata = ModelMetadata(
            model_id=model_id,
            iteration=iteration,
            architecture_name=network._architecture_name,
            architecture_params=network._architecture_params,
            export_timestamp=datetime.now().isoformat(),
            is_benchmark_breaker=is_benchmark_breaker,
            run_name=self.run_name,
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(
            f"Exported model {model_id} (architecture: {network._architecture_name}) to {model_path}"
        )
        return model_id

    @staticmethod
    def _validate_exportable(network: torch.nn.Module) -> None:
        if not hasattr(network, "_architecture_name") or not hasattr(
            network, "_architecture_params"
        ):
            raise ValueError(
                "Model must have _architecture_name and _architecture_params attributes for export"
            )


def create_model_from_architecture(architecture_name: str, **kwargs) -> torch.nn.Module:
    """Create model instance from architecture name and params."""
    if architecture_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture_name}. Known architectures: {', '.join(sorted(ARCHITECTURE_REGISTRY))}"
        )
    return ARCHITECTURE_REGISTRY[architecture_name](**kwargs)


def load_any_model(model_dir: str, model_id: str, device: str = "cpu") -> torch.nn.Module:
    """Load model from any directory without knowing architecture beforehand."""
    metadata_path = os.path.join(model_dir, f"{model_id}.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata for model {model_id} not found in {model_dir}")

    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)
    metadata = ModelMetadata.from_dict(metadata_dict)

    model_path = os.path.join(model_dir, f"{model_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights {model_id} not found in {model_dir}")

    model = create_model_from_architecture(
        metadata.architecture_name, **metadata.architecture_params
    )
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            # Usuwamy pierwsze 10 znaków ("_orig_mod.")
            new_key = key[10:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def get_models_from_directory(model_dir: str) -> List[Dict[str, Any]]:
    """Get list of all models from a directory by scanning for .json files."""
    models: List[Dict[str, Any]] = []

    if not os.path.exists(model_dir):
        return models

    for filename in os.listdir(model_dir):
        if not filename.endswith(".json"):
            continue

        metadata_path = os.path.join(model_dir, filename)
        try:
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue  # Skip corrupted files

        metadata = ModelMetadata.from_dict(metadata_dict)
        models.append(metadata.to_dict())

    models.sort(key=lambda x: x.get("iteration", 0))
    return models
