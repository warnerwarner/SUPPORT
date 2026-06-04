#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=3-00:00:00
#SBATCH --job-name=SUPPORT_DDP_4gpu
#SBATCH --output=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.log
#SBATCH --error=/gpfs/home/warnet02/shohamlab/tom/tmp/%j.err
#SBATCH --mail-type=END,FAIL,BEGIN
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
echo "Total tasks (GPUs): $SLURM_NTASKS"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="
echo ""

# Training parameters
data_dir="/gpfs/home/warnet02/data/stephen/run012"
n_epochs=20
exp_name="stephen_voltage_ddp_4gpu"
checkpoint_interval=1

# Change to project directory
cd /gpfs/data/shohamlab/tom/support

# Launch distributed training with srun
srun /gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python \
    -u -m src.train_distributed \
    --exp_name "$exp_name" \
    --is_folder \
    --noisy_data "$data_dir" \
    --n_epochs "$n_epochs" \
    --batch_size 16 \
    --patch_size 1 16 320 \
    --patch_interval 1 4 160 \
    --checkpoint_interval "$checkpoint_interval" \
    --depth 8 \
    --bs_size 1 3 \
    --is_raw \
    --use_amp

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
