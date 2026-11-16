#!/usr/bin/env python3
"""
REEF defensibility pilot.

One harness, one data loader, one fold generator for every arm.
Fixes vs. the original benchmark:
  * no checkpoint selection on a single validation sample: every neural arm
    trains for a fixed number of steps and the FINAL model predicts;
  * every arm trains on all non-test samples of the fold;
  * LASER sees exactly the same matrices as REEF (filtered microbiome);
  * LASER is trained full-batch for the same number of optimizer steps.

REEF / NVFlare-VFL use the original classes from src/nvflare_vfl unchanged,
except that `evaluate()` returns a monotone dummy score during training so the
original "best checkpoint" logic always keeps the latest round.

Outputs one JSON line per (scheme, seed, split, fold) task with per-sample
test probabilities for every arm. All metrics are computed afterwards.
"""
import os
import sys
import json
import time
import random
import argparse
import logging
import itertools
import traceback

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = "/home/sam/reef-exps/REEF"
sys.path.insert(0, os.path.join(ROOT, "LASER-VFL-master"))
sys.path.insert(0, ROOT)

logging.disable(logging.CRITICAL)

from src.nvflare_vfl.nvflare_native_trainer import (  # noqa: E402
    NVFlareNativeVFLTrainer,
    DomainAwareNVFlareTrainer,
    NVFlareVFLConfig,
    DomainAwareNVFlareConfig,
    load_coral_data,
)
from benchmarks.adapters.laser_adapter import LASERAdapter  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

N_ROUNDS = 50          # paper: 50 rounds for every method
WARMUP = 20            # paper: 20 warmup + 30 refinement
TOPK = 500
EXPERT_RATIOS = {"proteomics": 0.5, "metabolomics": 1.0, "microbiome": 0.8, "transcriptomics": 0.3}
EXPERT_WEIGHTS = {"transcriptomics": 1.5, "proteomics": 1.0, "metabolomics": 0.8, "microbiome": 0.5}
EQUAL_RATIOS = {k: 0.65 for k in EXPERT_RATIOS}
EQUAL_WEIGHTS = {k: 1.0 for k in EXPERT_WEIGHTS}


# ---------------------------------------------------------------------------
# Fixed-length training: original classes, dummy validation score
# ---------------------------------------------------------------------------
class FixedRoundsMixin:
    def evaluate(self, indices, return_predictions=False):
        if not return_predictions:
            # monotone in round -> original code always keeps the latest model
            return {"auroc": float(self.current_round), "accuracy": 0.0}
        return super().evaluate(indices, return_predictions=True)


class VFLFixed(FixedRoundsMixin, NVFlareNativeVFLTrainer):
    pass


class REEFFixed(FixedRoundsMixin, DomainAwareNVFlareTrainer):
    selector = "saliency"
    selector_seed = 0

    def _compute_domain_importance(self, train_indices):
        if self.selector == "saliency":
            return super()._compute_domain_importance(train_indices)
        rng = np.random.RandomState(self.selector_seed)
        y = self.labels[train_indices]
        out = {}
        for name, silo in self.silos.items():
            if self.selector == "random":
                out[name] = rng.rand(silo["n_features"])
            elif self.selector == "ttest":
                x = silo["features"][train_indices].numpy()
                a, b = x[y == 1], x[y == 0]
                se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b)) + 1e-8
                out[name] = np.abs(a.mean(0) - b.mean(0)) / se
            else:
                raise ValueError(self.selector)
        return out


def predict(trainer, test_idx):
    return trainer.evaluate(np.asarray(test_idx), return_predictions=True)["predictions"].astype(float).tolist()


def run_vfl(data, y, tr, te, seed):
    cfg = NVFlareVFLConfig(n_rounds=N_ROUNDS, seed=seed, device="cpu", patience=10**9)
    t = VFLFixed(data, y, cfg)
    t.train(np.asarray(tr), np.asarray(tr))
    return predict(t, te), None


def run_reef(data, y, tr, te, seed, ratios, weights, selector):
    cfg = DomainAwareNVFlareConfig(
        n_rounds=N_ROUNDS, seed=seed, device="cpu", patience=10**9,
        topk_features=TOPK, domain_warmup_rounds=WARMUP,
        omic_topk_ratios=dict(ratios), omic_biological_weights=dict(weights),
    )
    t = REEFFixed(data, y, cfg)
    t.selector = selector
    t.selector_seed = seed * 7919 + int(sum(te)) * 31 + len(tr)
    t.train(np.asarray(tr), np.asarray(tr))
    masks = {k: np.flatnonzero(s["feature_mask"].numpy()).tolist() for k, s in t.silos.items()}
    return predict(t, te), masks


def run_laser(data, y, tr, te, seed, masks=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if masks is not None:
        data = {k: v[:, masks[k]] for k, v in data.items()}
    names = list(data.keys())
    cfg = {"hidden_dim": 128, "embedding_dim": 64, "lr": 1e-3, "weight_decay": 1e-5}
    a = LASERAdapter(data=data, labels=y, omic_names=names, config=cfg, device=torch.device("cpu"))
    tr = np.asarray(tr)
    a._fit_scalers(tr)
    a._transform_data()
    a._create_model()
    a._create_optimizer()
    a._create_criterion()
    loader = a.create_dataloader(tr, batch_size=len(tr), shuffle=True, p_miss=0.0)
    for _ in range(N_ROUNDS):  # full batch -> 1 optimizer step per epoch
        a.train_epoch(loader)
    res = a.evaluate(a.create_dataloader(np.asarray(te), batch_size=len(te), shuffle=False), return_predictions=True)
    return res["predictions"].astype(float).tolist(), None


def run_enet(data, y, tr, te):
    tr, te = np.asarray(tr), np.asarray(te)
    X = np.concatenate([data[k] for k in sorted(data)], axis=1)
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=5000, tol=1e-4, random_state=0)
    m.fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1].tolist(), None


ARMS = [
    ("vfl_full", lambda d, y, tr, te, s, ctx: run_vfl(d, y, tr, te, s)),
    ("reef", lambda d, y, tr, te, s, ctx: run_reef(d, y, tr, te, s, EXPERT_RATIOS, EXPERT_WEIGHTS, "saliency")),
    ("reef_equal", lambda d, y, tr, te, s, ctx: run_reef(d, y, tr, te, s, EQUAL_RATIOS, EQUAL_WEIGHTS, "saliency")),
    ("reef_random", lambda d, y, tr, te, s, ctx: run_reef(d, y, tr, te, s, EXPERT_RATIOS, EXPERT_WEIGHTS, "random")),
    ("equal_random", lambda d, y, tr, te, s, ctx: run_reef(d, y, tr, te, s, EQUAL_RATIOS, EQUAL_WEIGHTS, "random")),
    ("equal_ttest", lambda d, y, tr, te, s, ctx: run_reef(d, y, tr, te, s, EQUAL_RATIOS, EQUAL_WEIGHTS, "ttest")),
    ("laser_full", lambda d, y, tr, te, s, ctx: run_laser(d, y, tr, te, s)),
    ("laser_reefsel", lambda d, y, tr, te, s, ctx: run_laser(d, y, tr, te, s, ctx["reef"])),
    ("enet", lambda d, y, tr, te, s, ctx: run_enet(d, y, tr, te)),
]


def make_tasks(y, scheme, seeds, n_repeats):
    n = len(y)
    tasks = []
    if scheme == "loocv":
        for s in seeds:
            for i in range(n):
                tasks.append({"scheme": scheme, "seed": s, "split": 0, "fold": i,
                              "train": [j for j in range(n) if j != i], "test": [i]})
    elif scheme == "rskf3":
        # split repeats are independent of the training seed -> seed vs split variance
        for r in range(n_repeats):
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1000 + r)
            folds = list(skf.split(np.zeros(n), y))
            for s in seeds:
                for f, (tr, te) in enumerate(folds):
                    tasks.append({"scheme": scheme, "seed": s, "split": r, "fold": f,
                                  "train": tr.tolist(), "test": te.tolist()})
    return tasks


_DATA = None


def _init_worker():
    global _DATA
    torch.set_num_threads(1)
    logging.disable(logging.CRITICAL)
    data, y, _ = load_coral_data(os.path.join(ROOT, "data"), cv_mode="loocv")
    _DATA = (data, y)


def run_task(task, arms, permute_seed=None):
    data, y = _DATA
    if permute_seed is not None:
        y = np.random.RandomState(permute_seed).permutation(y)
    out = dict(task)
    out["labels"] = y[task["test"]].astype(int).tolist()
    out["arms"] = {}
    ctx = {}
    for name, fn in ARMS:
        if name not in arms:
            continue
        if name == "laser_reefsel" and "reef" not in ctx:
            continue
        if name == "enet" and task["seed"] != 0:
            continue  # deterministic: one fit per split is enough
        t0 = time.time()
        try:
            probs, masks = fn(data, y, task["train"], task["test"], task["seed"], ctx)
            if masks is not None and name == "reef":
                ctx["reef"] = masks
            out["arms"][name] = {"p": probs, "sec": round(time.time() - t0, 2)}
            if name in ("reef", "reef_equal", "equal_ttest") and masks is not None:
                out["arms"][name]["mask"] = masks
        except Exception:
            out["arms"][name] = {"error": traceback.format_exc()[-800:]}
    if permute_seed is not None:
        out["permute_seed"] = permute_seed
    return out


def _run(args):
    return run_task(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["loocv", "rskf3"], required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--arms", default=",".join(a for a, _ in ARMS))
    ap.add_argument("--permutations", type=int, default=0, help="label-permutation runs (seed 0 only)")
    ap.add_argument("--workers", type=int, default=70)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    _init_worker()
    y = _DATA[1]
    arms = a.arms.split(",")
    seeds = list(range(a.seeds))
    if a.permutations:
        base = make_tasks(y, a.scheme, [0], a.repeats)
        jobs = [(t, arms, p + 1) for p in range(a.permutations) for t in base]
    else:
        jobs = [(t, arms, None) for t in make_tasks(y, a.scheme, seeds, a.repeats)]
    if a.limit:
        jobs = jobs[: a.limit]

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init_worker) as pool, open(a.out, "a") as fh:
        for i, res in enumerate(pool.imap_unordered(_run, jobs, chunksize=1)):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(jobs)} tasks, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} tasks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
