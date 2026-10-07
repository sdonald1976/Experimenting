"""Observation and inference storage (REQUIRED by the architecture).

Observation store: what was actually perceived. Append-only; arrays are made
read-only and hashed on arrival so the harness can verify they were never
altered.

Inference store: everything the learner derived. Every record carries:
    id, tick (when the inference was made), kind, mechanism tag
    (slot:name@version#params_hash), inputs (ids of observations / earlier
    inferences it was derived from), epistemic state, payload.
Records are never edited or deleted. A later revision is a NEW record that
references the old one (Q9, Q11).

Epistemic states (decision D1, option (a), see ASSUMPTIONS.md):
    KNOWN    - only raw observations. No inference is ever KNOWN.
    BELIEVED - an inference with some degree of support.
    UNKNOWN  - the learner explicitly does not know (unresolved).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

KNOWN, BELIEVED, UNKNOWN = "KNOWN", "BELIEVED", "UNKNOWN"


class ObservationStore:
    def __init__(self):
        self._obs: dict[str, dict] = {}
        self.hashes: dict[str, str] = {}

    def add(self, obs: dict) -> str:
        oid = f"obs:{obs['tick']}"
        frozen = {}
        h = hashlib.sha256()
        for k in ("left", "right"):
            a = np.array(obs[k], copy=True)
            a.setflags(write=False)
            frozen[k] = a
            h.update(a.tobytes())
        frozen["tick"] = int(obs["tick"])
        frozen["self_motion"] = dict(obs["self_motion"])
        h.update(repr(sorted(frozen["self_motion"].items())).encode())
        self._obs[oid] = frozen
        self.hashes[oid] = h.hexdigest()
        return oid

    def get(self, oid: str) -> dict:
        return self._obs[oid]

    def __len__(self):
        return len(self._obs)


@dataclass
class Inference:
    id: str
    tick: int
    kind: str
    mechanism: str
    inputs: list
    epistemic: str
    payload: dict = field(default_factory=dict)


class InferenceStore:
    def __init__(self):
        self.records: list[Inference] = []
        self.by_id: dict[str, Inference] = {}
        self._n = 0

    def add(self, tick, kind, mechanism, inputs, epistemic, **payload) -> Inference:
        if epistemic == KNOWN:
            raise ValueError("No inference may be KNOWN (D1 option a)")
        if epistemic not in (BELIEVED, UNKNOWN):
            raise ValueError(epistemic)
        if not mechanism or not inputs:
            raise ValueError("inference without provenance")
        self._n += 1
        rec = Inference(f"inf:{self._n}", int(tick), kind, mechanism, list(inputs), epistemic, payload)
        self.records.append(rec)
        self.by_id[rec.id] = rec
        return rec

    def audit(self) -> dict:
        bad = [r.id for r in self.records if not r.mechanism or not r.inputs
               or r.epistemic not in (BELIEVED, UNKNOWN)]
        kinds: dict[str, int] = {}
        for r in self.records:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        return {"n": len(self.records), "missing_provenance": bad, "by_kind": kinds}
