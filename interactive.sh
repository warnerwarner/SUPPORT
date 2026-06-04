#!/bin/bash
#SBATCH --partition=gl40s_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=INTERACTIVE
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err

sleep 3600
