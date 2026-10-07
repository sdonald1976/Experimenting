"""Developmental history of a persistent unit's evidence usefulness.

python -m harness.inspect_unit --split test --seed 2000                 # list mapped units
python -m harness.inspect_unit --split test --seed 2000 --unit 7        # full history of unit 7
    [--variant A_learned] [--kind location_self_frame]

Each line: tick (phase), evidence kind, usefulness old -> new, window sample counts, and the sample
that caused the change: W (same, from continuity) or B (different, from co-observation), its
distance, the record that produced it and the record it was matched to.
The thing name is harness ground truth shown for the human reader; the learner never had it.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="A_learned")
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--kind", default=None)
    a = ap.parse_args()
    d = os.path.join(ROOT, "results", a.split)
    h = json.load(open(os.path.join(d, "histories", f"seed{a.seed}_{a.variant}.json")))
    res = json.load(open(os.path.join(d, f"seed{a.seed}.json")))
    units = {str(u["unit"]): u for u in res["variants"][a.variant]["units"]}
    T2 = h["phase2_start_tick"]
    if a.unit is None:
        print(f"seed {a.seed} {a.variant}: phase 2 starts at tick {T2}. Mapped units:")
        for uid, u in units.items():
            print(f"  unit {uid:>4}  thing={h['units'][uid]['thing']:<28} records={u['n_records']:>4} "
                  f"created t={u['created_tick']:<4} motion={u['motion_class']:<6} "
                  f"final={ {k[:8]: round(v, 2) for k, v in u['learned_final'].items()} } "
                  f"oracle={ {k[:8]: round(v, 2) for k, v in u['oracle'].items()} }")
        return
    u = h["units"][str(a.unit)]
    print(f"unit {a.unit} ({u['thing']}, ground truth for the reader only); phase 2 starts at tick {T2}")
    for tick, kind, old, new, nW, nB, causes in u["history"]:
        if a.kind and kind != a.kind:
            continue
        if causes and not isinstance(causes[0], (list, tuple)):
            causes = [causes]
        desc = "; ".join(f"{c[0]}({c[4] if len(c) > 4 else 'short'}) d={c[1]} {c[2]} vs {c[3]}" for c in causes[:4])
        more = f" (+{len(causes) - 4} more)" if len(causes) > 4 else ""
        print(f"  t={tick:<4}(P{1 if tick < T2 else 2}) {kind:<32} {old:.3f} -> {new:.3f}   W={nW:<4} B={nB:<5} "
              f"caused by {len(causes)} sample(s): {desc}{more}")


if __name__ == "__main__":
    main()
