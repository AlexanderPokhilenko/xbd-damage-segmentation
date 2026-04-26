from typing import Callable, Dict

import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    return sorted(_MODEL_REGISTRY.keys())
