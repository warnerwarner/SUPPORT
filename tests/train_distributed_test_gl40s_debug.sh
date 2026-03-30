#!/bin/bash
#SBATCH --partition=gl40s_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=1:00:00
#SBATCH --job-name=SUPPORT_DDP_TEST_DEBUG
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

# Change to project directory
cd /gpfs/data/shohamlab/tom/support

# Launch distributed training across all nodes
# srun automatically distributes tasks across nodes

srun /gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python \
    -u -c "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
