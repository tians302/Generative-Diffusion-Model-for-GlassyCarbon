#!/bin/bash
#PBS -A FoundMLIP
#PBS -q capacity
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -N gc-diffusion
cd /lus/grand/projects/QuantMatManufact/tians/diffusion-model
source .venv/bin/activate
python train.py
