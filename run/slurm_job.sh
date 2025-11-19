#!/bin/bash
#SBATCH --partition=cpu-single
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=40G
#SBATCH --job-name=matched_filter
#SBATCH --output=slurm/matched_filter/out_%j.out
#SBATCH --error=slurm/matched_filter/out_%j.out
python3 ~/sds/software/matched_filter/run/matched_filter.py
