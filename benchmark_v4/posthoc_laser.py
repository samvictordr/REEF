#!/usr/bin/env python3
"""Add LASER selection arms to an existing pilot jsonl, reusing the stored REEF masks."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_v3 as P
import multiprocessing as mp

ARMS = ["laser_salQ", "laser_rndE", "laser_rndQ", "laser_ttQ"]


def work(row):
    data, y = P._DATA
    tr, te, s = row["train"], row["test"], row["seed"]
    ctx = {"reef_equal": row["arms"]["reef_equal"]["mask"]}
    out = {k: row[k] for k in ("scheme", "seed", "split", "fold", "train", "test", "labels")}
    out["arms"] = {}
    for name, fn in P.ARMS:
        if name in ARMS:
            out["arms"][name] = {"p": fn(data, y, tr, te, s, ctx)[0]}
    return out


if __name__ == "__main__":
    src, dst, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = [json.loads(l) for l in open(src)]
    with mp.get_context("fork").Pool(workers, initializer=P._init_worker) as pool, open(dst, "w") as fh:
        for i, r in enumerate(pool.imap_unordered(work, rows)):
            fh.write(json.dumps(r) + "\n")
    print("done", len(rows))
