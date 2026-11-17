#!/usr/bin/env bash
# Runs after rskf3 finishes: permutation test, then training-length / LASER-lr sensitivity.
cd /home/sam/reef-exps/pilot
export TMPDIR=/home/sam/reef-exps/tmp MPLCONFIGDIR=/home/sam/reef-exps/tmp WANDB_DIR=/home/sam/reef-exps/tmp CUDA_VISIBLE_DEVICES=
PY=../venv/bin/python
while pgrep -f "[p]ilot_v2.py --scheme rskf3" >/dev/null; do sleep 30; done
$PY pilot_v3.py --scheme rskf3 --repeats 1 --permutations 100 --arms vfl_full,reef,equal_random,laser_full,laser_rndQ,enet --workers 74 --out perm.jsonl > perm.log 2>&1
for R in 20 100; do
  PILOT_ROUNDS=$R $PY pilot_v3.py --scheme loocv --seeds 10 --arms vfl_full,reef,equal_random,laser_full,laser_rndQ --workers 74 --out sens_r$R.jsonl > sens_r$R.log 2>&1
done
PILOT_LASER_LR=1e-4 $PY pilot_v3.py --scheme loocv --seeds 10 --arms laser_full,laser_rndQ --workers 74 --out sens_laserlr1e-4.jsonl > sens_lr.log 2>&1
$PY posthoc_laser.py rskf3.jsonl rskf3_laser.jsonl 74 > posthoc_rskf3.log 2>&1
echo ALLDONE > queue2.done
