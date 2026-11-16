# Benchmark harness for preprint v4 (arXiv 2601.00050v4)

These scripts produce every number, table and figure in v4,
"What makes vertical federated learning work with thirteen samples?
A controlled benchmark on coral thermal-stress multi-omics".
v4 supersedes v1–3. The checkpoint-selection bug behind the v1–3 results is described in the v4 Introduction.

## What changed relative to `benchmarks/run_comprehensive_benchmark.py`

- No checkpoint selection on a single validation sample: every neural arm trains for a fixed number of steps and the final model predicts.
- Every arm trains on all non-test samples of the fold.
- LASER-VFL receives the same matrices as the other arms (filtered microbiome, 12,432 features), trained full-batch for the same number of optimiser steps.
- The NVFlare VFL baseline and the saliency pipeline (`DomainAwareNVFlareTrainer`, called REEF in v1–3) use the original classes in `src/nvflare_vfl` unchanged. The only override is `evaluate()`, which returns a monotone dummy score during training so the original best-checkpoint logic always keeps the latest round.

## Files

| File | Purpose |
|---|---|
| `pilot_v2.py` | Main harness. Arms `vfl_full`, `reef`, `reef_equal`, `reef_random`, `equal_random`, `equal_ttest`, `laser_full`, `laser_reefsel`, `enet`; schemes `loocv` and `rskf3`. The elastic net is fitted for seed 0 only, since it is deterministic. |
| `pilot_v3.py` | `pilot_v2.py` plus the LASER mask arms (`laser_salQ`, `laser_rndE`, `laser_rndQ`, `laser_ttQ`), label permutations (`--permutations`) and sensitivity variables `PILOT_ROUNDS`, `PILOT_LASER_LR` and `PILOT_EXCLUDE`. |
| `posthoc_laser.py` | Adds the LASER mask arms to an existing run, using the stored saliency masks. |
| `queue2.sh`, `queue3.sh` | The exact run sequences: permutation test, training-length and LASER learning-rate sensitivity, and exclusion of the two possibly mislabelled samples. |
| `analyze.py` | Means, bootstrap CIs, seed-vs-split SD, Nadeau–Bengio corrected t-test, Brown–Forsythe test and Nogueira selection stability. Merge files with `+`, e.g. `python analyze.py loocv.jsonl+loocv_laser.jsonl`. |
| `figures.py` | Figures 1–5 and `figs/summary.json`. |

## Running

The scripts were run from `/home/sam/reef-exps/pilot/` with the repository at `/home/sam/reef-exps/REEF` (`ROOT` in the harness; edit it for your checkout).
Data come from Zenodo record 6861688 (`Supplemental_Data.xlsx`), extracted with `scripts/extract_excel.py --auto --out data/processed/`.
LASER-VFL is the authors' code at commit `b7ff159`.
Its upstream `data/`, `models/lstm.py` and `models/mimic_model_utils.py` must be present in `LASER-VFL-master/`.
Environment used: Python 3.10, torch 2.5.1 (CPU), numpy 1.26.4, scikit-learn 1.5.2, scipy 1.13.1, nvflare 2.7.1.

```bash
python pilot_v2.py --scheme loocv --seeds 20 --workers 74 --out loocv.jsonl
python pilot_v2.py --scheme rskf3 --seeds 10 --repeats 10 --workers 74 --out rskf3.jsonl
python posthoc_laser.py loocv.jsonl loocv_laser.jsonl 74
python posthoc_laser.py rskf3.jsonl rskf3_laser.jsonl 74
./queue2.sh && ./queue3.sh
python analyze.py loocv.jsonl+loocv_laser.jsonl rskf3.jsonl+rskf3_laser.jsonl
python figures.py
```

The LOOCV run used the first harness revision, which differs from `pilot_v2.py` only in fitting the elastic net for every seed rather than seed 0.

## Per-sample predictions

Each run writes one JSON line per (scheme, seed, split, fold) task.
Each line holds the fold assignment, the test labels, every arm's per-sample test probabilities and, for selection arms, the selected feature indices.
The prediction files (`loocv*.jsonl`, `rskf3*.jsonl`, `perm.jsonl`, `sens_*.jsonl`, `excl2.jsonl`) will be added to `benchmark_v4/results/`.
