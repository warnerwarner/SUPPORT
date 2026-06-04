#!/bin/bash
#SBATCH --partition=shohamlab # partition name
#SBATCH --nodes=1  # number of nodes
#SBATCH --ntasks-per-node=1  # number of tasks per node
#SBATCH --cpus-per-task=2  # number of CPUs per task
#SBATCH --mem=64G  # active memory
#SBATCH --gres=gpu:1  # number of GPUs per node
#SBATCH --time=24:00:00  # overall availability time
#SBATCH --job-name=SUPPORT  # job name
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log  # log output file
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err  # error output file
#SBATCH --mail-type=END,FAIL,BEGIN  # email notifications; type: BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=tom.warner@nyulangone.org  # email address to send notifications to

file_name="/gpfs/data/shohamlab/tom/stephen/flat_high-power-FOV4_force1s_VC_440Hz_9X_1p4x1SAM_0p59t-0p8s-FF_0p25ms-fb_bin1_EOD-on_00002_reconstructed_filtered.tif"  
module load miniconda3/gpu/4.9.2
conda activate SUPPORT
cd data/support

# python -m src.train --exp_name stephenVoltage_bs1_if_881 --noisy_data "$file_name" --checkpoint_interval 1 --n_epochs 11 --bs_size 1 1 --input_frames 881 --patch_size 881 128 128
# python -m src.train --exp_name stephenVoltage_bs5_if_881 --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 5 5 --input_frames 881 --patch_size 881 128 128
# python -m src.train --exp_name stephenVoltage_bs10_if_881 --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 10 10 --input_frames 881 --patch_size 881 128 128
# python -m src.train --exp_name stephenVoltage_bs20_if_881 --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 20 20 --input_frames 881 --patch_size 881 128 128

python -m src.train --exp_name stephenVoltage_bs1_rolling --noisy_data "$file_name" --checkpoint_interval 1 --n_epochs 11 --bs_size 1 1 --is_folder
# python -m src.train --exp_name stephenVoltage_bs5_rolling --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 5 5  --is_folder
# python -m src.train --exp_name stephenVoltage_bs10_rolling --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 10 10  --is_folder
# python -m src.train --exp_name stephenVoltage_bs20_rolling --noisy_data "$file_name" --checkpoint_interval 5 --n_epochs 11 --bs_size 20 20 --is_folder
