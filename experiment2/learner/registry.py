"""Mechanism registry.

Every mechanism that is not required by the architecture is a subclass of
`Mechanism`, carries status EXPERIMENTAL, a version, its parameters and its
assumptions in plain words. All of it is logged with every run, and every
inference a mechanism produces records (name, version, params_hash).

Swapping a mechanism = choosing a different registered class in the learner
config. Nothing else in the learner refers to concrete classes.
"""
from __future__ import annotations

import hashlib
import json

REGISTRY: dict[str, dict[str, type]] = {}


def register(slot: str):
    def deco(cls):
        REGISTRY.setdefault(slot, {})[cls.NAME] = cls
        cls.SLOT = slot
        return cls
    return deco


class Mechanism:
    SLOT = "?"
    NAME = "?"
    VERSION = "0"
    STATUS = "EXPERIMENTAL"
    DEFAULTS: dict = {}
    ASSUMPTIONS: list[str] = []

    def __init__(self, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{self.NAME}: unknown params {unknown}")
        self.p = {**self.DEFAULTS, **params}
        self.params_hash = hashlib.sha1(json.dumps(self.p, sort_keys=True).encode()).hexdigest()[:10]

    def describe(self) -> dict:
        return {"slot": self.SLOT, "name": self.NAME, "version": self.VERSION,
                "status": self.STATUS, "params": self.p, "params_hash": self.params_hash,
                "assumptions": self.ASSUMPTIONS}

    @property
    def tag(self) -> str:
        return f"{self.SLOT}:{self.NAME}@{self.VERSION}#{self.params_hash}"


def build(slot: str, spec: dict) -> Mechanism:
    cls = REGISTRY[slot][spec["name"]]
    return cls(**spec.get("params", {}))
