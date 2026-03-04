#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=SUPPORT_DDP_TRAIN
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=tom.warner@nyulangone.org,9145136289@tmomail.net,9145136289@mailmymobile.net

# Get the master node hostname (first node in allocation)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=========================================="
echo "Multi-Node Distributed Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks (GPUs): $SLURM_NTASKS"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="
echo ""

# Training parameters
data_dir="/gpfs/home/warnet02/data/stephen"
n_epochs=20
exp_name="stephenvoltage_if_61_bs_2_2_l107l203"
checkpoint_interval=5  # Match original training (save every 5 epochs)

# Change to project directory
cd /gpfs/data/shohamlab/tom/support

# Launch distributed training across all nodes
# srun automatically distributes tasks across nodes

srun /gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python \
    -u -m src.train_distributed \
    --exp_name "$exp_name" \
    --is_folder \
    --noisy_data "$data_dir" \
    --n_epochs "$n_epochs" \
    --batch_size 16 \
    --patch_size 61 16 320 \
    --patch_interval 10 4 160 \
    --checkpoint_interval "$checkpoint_interval" \
    --depth 8 \
    --bs_size 2 2 \
    --is_raw \
    --is_zarr \
    --training_size 20 \
    --use_amp \
    --n_cpu 7 \
    --loss_coef 0.7 0.3

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
