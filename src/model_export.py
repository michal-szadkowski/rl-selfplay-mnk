import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import torch
from .alg.ppo import ActorCriticModule


class ModelExporter:
    def __init__(self, run_name: str = None, base_dir: str = "models"):
        self.run_name = run_name or self._generate_timestamp()
        self.export_dir = os.path.join(base_dir, self.run_name)
        os.makedirs(self.export_dir, exist_ok=True)

    def _generate_timestamp(self) -> str:
        """Generate timestamp in format YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_model(self, network: torch.nn.Module, iteration: int,
                    is_benchmark_breaker: bool = False) -> str:
        """Export a model with automatic architecture detection"""
        # Auto-detect architecture from model instance
        if not hasattr(network, '_architecture_name') or not hasattr(network, '_architecture_params'):
            raise ValueError("Model must have _architecture_name and _architecture_params attributes for export")

        model_id = f"model_{iteration:05d}"
        model_file = f"{model_id}.pt"
        metadata_file = f"{model_id}.json"

        model_path = os.path.join(self.export_dir, model_file)
        metadata_path = os.path.join(self.export_dir, metadata_file)

        # Save model weights
        torch.save(network.state_dict(), model_path)

        # Save metadata with architecture info
        metadata = {
            "model_id": model_id,
            "iteration": iteration,
            "architecture": {
                "name": network._architecture_name,
                "params": network._architecture_params
            },
            "export_timestamp": datetime.now().isoformat(),
            "is_benchmark_breaker": is_benchmark_breaker,
            "run_name": self.run_name
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Exported model {model_id} (architecture: {network._architecture_name}) to {model_path}")
        return model_id

    def load_model(self, model_id: str, device: str = "cpu") -> torch.nn.Module:
        """Load a specific exported model"""
        # Load metadata
        metadata = self.get_model_metadata(model_id)
        if metadata is None:
            raise FileNotFoundError(f"Metadata for model {model_id} not found")

        # Create model from architecture info
        arch_info = metadata["architecture"]
        model = create_model_from_architecture(arch_info["name"], **arch_info["params"])

        # Load weights
        model_file = f"{model_id}.pt"
        model_path = os.path.join(self.export_dir, model_file)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model"""
        metadata_file = f"{model_id}.json"
        metadata_path = os.path.join(self.export_dir, metadata_file)

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)


def create_model_from_architecture(architecture_name: str, **kwargs) -> torch.nn.Module:
    """Create model instance from architecture name and params"""
    if architecture_name == "actor_critic":
        return ActorCriticModule(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}. Known architectures: actor_critic")


def load_any_model(model_dir: str, model_id: str, device: str = "cpu") -> torch.nn.Module:
    """Load model from any directory without knowing architecture beforehand"""
    metadata_path = os.path.join(model_dir, f"{model_id}.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata for model {model_id} not found in {model_dir}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create model from architecture
    arch_info = metadata["architecture"]
    model = create_model_from_architecture(arch_info["name"], **arch_info["params"])

    # Load weights
    model_path = os.path.join(model_dir, f"{model_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights {model_id} not found in {model_dir}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_models_from_directory(model_dir: str) -> List[Dict[str, Any]]:
    """Get list of all models from a directory by scanning for .json files"""
    models = []

    if not os.path.exists(model_dir):
        return models

    for filename in os.listdir(model_dir):
        if filename.endswith(".json"):
            model_id = filename[:-5]  # Remove .json
            metadata_path = os.path.join(model_dir, filename)

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
            except (json.JSONDecodeError, FileNotFoundError):
                continue  # Skip corrupted files

    # Sort by iteration
    models.sort(key=lambda x: x.get("iteration", 0))
    return models


def get_benchmark_breakers(model_dir: str) -> List[Dict[str, Any]]:
    """Get list of models that broke benchmark from a directory"""
    all_models = get_models_from_directory(model_dir)
    return [model for model in all_models if model.get("is_benchmark_breaker", False)]