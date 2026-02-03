#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=3-00:00:00
#SBATCH --job-name=SUPPORT_DDP_20gpu_wider
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=tom.warner@nyulangone.org

# Get the master node hostname (first node in allocation)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=========================================="
echo "Multi-Node Distributed Training - WIDER MODEL (1.5x)"
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
echo "MODEL CAPACITY INCREASE: 1.5x wider"
echo "  blind_conv_channels: 64 → 96"
echo "  unet_channels: [64,128,256,512,1024] → [96,192,384,768,1536]"
echo "  one_by_one_channels: [32,16] → [48,24]"
echo "  last_layer_channels: [64,32,16] → [96,48,24]"
echo "  Estimated parameters: ~4.5-5.5M (vs ~1.8M baseline)"
echo "=========================================="
echo ""

# Training parameters
data_dir="/gpfs/home/warnet02/data/stephen"
n_epochs=20
exp_name="stephenvoltage_if_61_bs_1_wider_1.5x_20epochs"
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
    --patch_interval 1 4 160 \
    --checkpoint_interval "$checkpoint_interval" \
    --depth 8 \
    --bs_size 1 1 \
    --is_raw \
    --training_size 10 \
    --use_amp \
    --n_cpu 7 \
    --blind_conv_channels 96 \
    --unet_channels 96 192 384 768 1536 \
    --one_by_one_channels 48 24 \
    --last_layer_channels 96 48 24

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
