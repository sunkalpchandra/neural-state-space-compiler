"""Lightweight name → class registry.

Every model component (encoder, decoder, dynamics, dataset, metric) registers
itself under a string key so the compiler can enumerate candidates without
hard-coded imports.

Example
-------
>>> DYNAMICS = Registry("dynamics")
>>> @DYNAMICS.register("linear")
... class Linear: ...
>>> DYNAMICS.build("linear", latent_dim=4)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, type[T]] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            k = key or cls.__name__
            if k in self._items and self._items[k] is not cls:
                raise KeyError(f"{self.name}: key '{k}' already registered to {self._items[k]}")
            self._items[k] = cls
            setattr(cls, "registry_key", k)
            return cls

        return deco

    def get(self, key: str) -> type[T]:
        if key not in self._items:
            raise KeyError(
                f"{self.name}: unknown key '{key}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def build(self, key: str, **kwargs: Any) -> T:
        return self.get(key)(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._items)


ENCODERS: Registry = Registry("encoders")
DECODERS: Registry = Registry("decoders")
DYNAMICS: Registry = Registry("dynamics")
SYSTEMS: Registry = Registry("systems")
DATASETS: Registry = Registry("datasets")
BASELINES: Registry = Registry("baselines")
