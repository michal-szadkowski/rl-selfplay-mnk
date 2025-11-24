import os
import glob
import json
from typing import List, Dict, Any, Set, Tuple
import torch

from utils.model_export import load_any_model


class ModelInfo:
    """Container for model information and metadata."""

    def __init__(self, model_id: str, model_path: str, metadata: Dict[str, Any]):
        self.model_id = model_id
        self.model_path = model_path
        self.metadata = metadata
        self.model = None  # Loaded on demand
        self.run_name = metadata.get("run_name", "unknown")
        self.iteration = metadata.get("iteration", 0)

    @property
    def unique_key(self) -> Tuple[str, int]:
        """Get unique identifier (run_name, iteration) for this model."""
        return (self.run_name, self.iteration)

    @property
    def unique_id(self) -> str:
        """Get string representation of unique identifier."""
        return f"{self.run_name}_{self.iteration:06d}"

    def load_model(self, device: str = "cpu") -> torch.nn.Module:
        """Load model if not already loaded."""
        if self.model is None:
            self.model = load_any_model(
                os.path.dirname(self.model_path), self.model_id, device
            )
        else:
            # Model exists in RAM, move to target device if needed
            if device != "cpu" and not next(self.model.parameters()).is_cuda:
                self.model = self.model.to(device)
        return self.model

    def unload_model(self, hard: bool = False) -> None:
        """Unload model from GPU memory to CPU RAM, or completely from RAM if hard=True."""
        if self.model is not None:
            # Move to CPU if on GPU to free VRAM but keep model in RAM
            if hasattr(self.model, "cuda") and next(self.model.parameters()).is_cuda:
                self.model.cpu()
            # Hard unload: completely remove from RAM
            if hard:
                del self.model
                self.model = None


class ModelLoader:
    """Flexible model loading from multiple sources."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load_from_paths(self, paths: List[str]) -> List[ModelInfo]:
        """Load models from multiple sources (files, directories, glob patterns)."""
        all_models = []
        seen_keys: Set[Tuple[str, int]] = set()

        for path in paths:
            models = self._load_from_single_path(path)
            for model in models:
                if model.unique_key not in seen_keys:
                    all_models.append(model)
                    seen_keys.add(model.unique_key)

        # Sort models by run_name then iteration for consistent ordering
        all_models.sort(key=lambda x: (x.run_name, x.iteration))
        return all_models

    def _load_from_single_path(self, path: str) -> List[ModelInfo]:
        """Load models from a single path (file, directory, or glob)."""
        path = os.path.expanduser(path)

        if os.path.isfile(path):
            return self._load_from_file(path)
        elif os.path.isdir(path):
            return self._load_from_directory(path)
        else:
            # Try glob pattern
            glob_matches = glob.glob(path)
            if glob_matches:
                models = []
                for match in glob_matches:
                    if os.path.isfile(match):
                        models.extend(self._load_from_file(match))
                    elif os.path.isdir(match):
                        models.extend(self._load_from_directory(match))
                return models
            else:
                return []

    def _load_from_file(self, file_path: str) -> List[ModelInfo]:
        """Load model from a specific .pt file."""
        if not file_path.endswith(".pt"):
            return []

        # Look for corresponding metadata file
        base_path = file_path[:-3]  # Remove .pt
        metadata_path = f"{base_path}.json"

        if not os.path.exists(metadata_path):
            return []

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            model_id = os.path.basename(base_path)
            return [ModelInfo(model_id, file_path, metadata)]

        except Exception:
            return []

    def _load_from_directory(self, dir_path: str) -> List[ModelInfo]:
        """Load all models from a directory."""
        models = []
        metadata_files = glob.glob(os.path.join(dir_path, "*.json"))

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                model_id = os.path.basename(metadata_file)[:-5]  # Remove .json
                model_path = os.path.join(dir_path, f"{model_id}.pt")

                if os.path.exists(model_path):
                    models.append(ModelInfo(model_id, model_path, metadata))

            except Exception:
                continue

        return models
