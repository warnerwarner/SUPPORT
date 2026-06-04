#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --job-name=SUPPORT_DDP_test
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tom.warner@nyulangone.org

# Set master address for communication
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="
echo ""

# Training parameters - MATCHING GOLD STANDARD EXACTLY
# This test uses isotropic dilations (like gold standard) with:
#   - is_raw=True for data loading/alignment
#   - Isotropic dilations in model (anisotropic code disabled)
#   - 10 files to match gold standard
#   - Should see loss ~0.60-0.64 at batch 1501
data_dir="/gpfs/home/warnet02/data/stephen"
n_epochs=1
exp_name="gold_standard_match_isotropic_10files_1epoch"
checkpoint_interval=5  # Match original training (save every 5 epochs, but we only run 1)

# Change to project directory
cd /gpfs/data/shohamlab/tom/support

# Launch distributed training with srun (automatically sets SLURM_PROCID, SLURM_LOCALID, etc.)
# Using same params as multi-node training but with only 2 files to test quickly
srun /gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python \
    -u -m src.train_distributed \
    --exp_name "$exp_name" \
    --is_folder \
    --noisy_data "$data_dir" \
    --n_epochs "$n_epochs" \
    --batch_size 16 \
    --patch_size 61 16 320 \
    --patch_interval 1 4 160 \
    --checkpoint_interval "$checkpoint_interval" \
    --depth 8 \
    --bs_size 1 1 \
    --is_raw \
    --training_size 10 \
    --use_amp \
    --n_cpu 10 \
    --logging_interval_batch 10  # Log more frequently for testing (default is 50)

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
