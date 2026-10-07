"""Learner process entry point. Runs in its own OS process; the only channel
to the outside is the pipe. Messages in: ("calib", dict), ("obs", dict),
("finish", None). Messages out: export dicts, then a final report."""
from __future__ import annotations

import time
import traceback


def learner_main(conn, config: dict):
    from .core import Learner  # imported inside the child process
    learner = None
    t_used = 0.0
    try:
        while True:
            kind, payload = conn.recv()
            if kind == "calib":
                learner = Learner(payload, config)
                conn.send(("ready", learner.describe()))
            elif kind == "obs":
                t0 = time.process_time()
                out = learner.step(payload)
                t_used += time.process_time() - t0
                conn.send(("export", out))
            elif kind == "finish":
                rep = learner.final_report()
                rep["cpu_seconds"] = t_used
                conn.send(("final", {"report": rep, "state_dump": learner.state_dump()}))
                break
    except Exception:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()
