#!/usr/bin/env bash
# After queue2: LOOCV without the two samples flagged as possibly mislabelled (Williams et al. 2023).
cd /home/sam/reef-exps/pilot
export TMPDIR=/home/sam/reef-exps/tmp MPLCONFIGDIR=/home/sam/reef-exps/tmp WANDB_DIR=/home/sam/reef-exps/tmp CUDA_VISIBLE_DEVICES=
while [ ! -f queue2.done ]; do sleep 30; done
PILOT_EXCLUDE=MC-289_T5-HiT_2998,MC-289_T5-Amb_1721 ../venv/bin/python pilot_v3.py --scheme loocv --seeds 10 --arms vfl_full,reef,equal_random,laser_full,laser_rndQ,enet --workers 74 --out excl2.jsonl > excl2.log 2>&1
echo ALLDONE > queue3.done
