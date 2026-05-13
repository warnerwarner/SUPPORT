# Testing Checklist for Distributed Training

Complete this checklist as you test each phase. Mark items with [X] when complete.

## Pre-Flight Checks ✓

- [X] Python syntax verified
- [X] SLURM scripts syntax verified  
- [X] Scripts are executable
- [X] NCCL backend available
- [X] Data directory exists: /gpfs/home/warnet02/data/stephen/run012

## Phase 1: Environment Validation (5 minutes)

```bash
cd /gpfs/data/shohamlab/tom/support
```

### Check Python environment
- [ ] Run: `/gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python -c "import torch.distributed as dist; print('NCCL:', dist.is_nccl_available())"`
- [ ] Expected output: `NCCL: True`

### Verify data exists
- [ ] Run: `ls -lh /gpfs/home/warnet02/data/stephen/run012/ | head -5`
- [ ] Expected: List of .tif files

### Check cluster availability
- [ ] Run: `sinfo -p gpu4_medium | grep idle`
- [ ] Note number of idle nodes: _______

## Phase 2: 2-GPU Test (~30-45 minutes)

**Goal:** Verify distributed training works at all

### Submit test job
```bash
sbatch train_distributed_test.sh
```

- [ ] Job submitted successfully
- [ ] Job ID: _______
- [ ] Job state is PD (pending) or R (running)

### Monitor job startup (first 5 minutes)
```bash
# Watch job queue
squeue -u warnet02

# Once running, check logs
tail -f /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log
```

**Critical checks:**
- [ ] "Initialized Distributed Training" appears
- [ ] World size: 2
- [ ] Rank: 0 and Rank: 1 mentioned
- [ ] Master addr shows node name (e.g., gn-0018)
- [ ] "Loaded ... Shape: [3000, 512, 512]" (or similar)
- [ ] "Dataset size: ..." appears
- [ ] Training starts (Epoch [0/2] Batch [1/...])

### Monitor during training
```bash
# Use the monitor script
./monitor_training.sh <JOBID>
```

- [ ] Both GPUs show high utilization (>85%)
- [ ] Memory usage similar on both GPUs
- [ ] Loss decreases over batches
- [ ] No error messages in logs

### Verify completion
```bash
# Check job finished
squeue -u warnet02

# Check final logs
tail -50 /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log

# Check checkpoints
ls -lh results/saved_models/ddp_2gpu_test/
```

- [ ] Job completed (no longer in queue)
- [ ] "Training complete!" message in log
- [ ] Checkpoints exist: model_0.pth, model_1.pth
- [ ] NO duplicate files from rank 1
- [ ] Final loss logged

### Record results
- Epoch 0 final loss: _______
- Epoch 1 final loss: _______
- Total runtime: _______ minutes
- Any errors? _______

**If ANY of the above failed, STOP and debug before proceeding!**

---

## Phase 3: 4-GPU Single-Node Test (~2 hours first epoch)

**Only proceed if Phase 2 succeeded completely!**

### Submit single-node job
```bash
sbatch train_distributed_single_node.sh
```

- [ ] Job ID: _______
- [ ] Job running on node: _______

### Startup validation (first 10 minutes)
```bash
tail -f /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log
```

- [ ] World size: 4
- [ ] All 4 ranks (0, 1, 2, 3) appear
- [ ] Effective batch size: 64 (16 × 4)
- [ ] Training starts

### GPU utilization check
```bash
# SSH to compute node
ssh <node_name>
watch -n 1 nvidia-smi
```

- [ ] All 4 GPUs show processes
- [ ] GPU utilization: ~90-100% on all
- [ ] Memory usage similar across GPUs

### Performance validation (after 1 hour)
```bash
# Check progress
./monitor_training.sh <JOBID>

# Estimate epoch time
# Note current batch number and time
```

- [ ] Batches/second: _______
- [ ] Estimated epoch time: _______ hours
- [ ] Expected: ~1.5-2 hours (vs ~6 hours single-GPU)
- [ ] Speedup: _______× 

### First epoch completion
```bash
tail -100 /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log | grep "Epoch \[0/500\] COMPLETE"
```

- [ ] Epoch 0 completed
- [ ] Final loss: _______
- [ ] Checkpoint saved
- [ ] Time for epoch 0: _______ hours

### Compare to single-GPU baseline
**Your previous single-GPU training:**
- Epoch 0 loss: ~0.565 (expected)
- Current 4-GPU epoch 0 loss: _______
- Difference: _______ (should be similar, ±0.05)

**If loss is very different (>0.1 difference), investigate before continuing!**

### Long-term monitoring
```bash
# Let it run for a few epochs, then check periodically
watch -n 300 './monitor_training.sh <JOBID>'
```

- [ ] Multiple epochs completed successfully
- [ ] Loss continues to decrease
- [ ] No crashes or hangs
- [ ] Checkpoints saving regularly

---

## Phase 4: Multi-Node Test (Optional: 2 nodes × 4 GPUs)

**Only if you want to validate multi-node before full 20-GPU run**

### Create 2-node test script
```bash
sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=4:00:00
#SBATCH --job-name=SUPPORT_2nodes_test

export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

cd /gpfs/data/shohamlab/tom/support

srun /gpfs/data/shohamlab/tom/voltage_imaging/.pixi/envs/default/bin/python \\
    -u -m src.train_distributed \\
    --exp_name ddp_2nodes_test \\
    --is_folder \\
    --noisy_data /gpfs/home/warnet02/data/stephen/run012 \\
    --n_epochs 2 \\
    --batch_size 16 \\
    --patch_interval 10 64 64 \\
    --checkpoint_interval 1 \\
    --use_amp
EOF
```

- [ ] Job ID: _______
- [ ] Nodes: _______

### Multi-node specific checks
```bash
tail -f /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log
```

- [ ] World size: 8
- [ ] Master node address shown
- [ ] All 8 ranks initialize
- [ ] No NCCL timeout errors
- [ ] Training progresses

### Verify both nodes active
```bash
# Get node list
squeue -j <JOBID> -o "%N"

# Check each node
ssh <node1> nvidia-smi
ssh <node2> nvidia-smi
```

- [ ] Node 1: 4 GPU processes active
- [ ] Node 2: 4 GPU processes active
- [ ] Both nodes show high utilization

### Cross-node communication test
```bash
# On one of the nodes during training
ssh <node1>
ifstat -i ib0 1
```

- [ ] Network traffic visible during training
- [ ] Bursts of traffic during gradient sync

---

## Phase 5: Full Production (5 nodes × 4 GPUs = 20 GPUs)

**Only proceed if all previous phases succeeded!**

### Final checks before launching
```bash
# Verify 5 nodes available
sinfo -p gpu4_medium | grep -E "idle|mix"

# Check disk space for checkpoints
df -h results/saved_models/
```

- [ ] At least 5 nodes available (idle or mix)
- [ ] Sufficient disk space (need ~50GB for checkpoints)

### Launch production training
```bash
sbatch train_distributed_multi_node.sh
```

- [ ] Job ID: _______
- [ ] Job submitted successfully
- [ ] Expected start time: _______

### Initial validation (first 30 minutes)
```bash
./monitor_training.sh <JOBID>
```

- [ ] World size: 20
- [ ] All 5 nodes listed
- [ ] Master node: _______
- [ ] All 20 ranks initialize
- [ ] Training starts

### Performance measurement (after 1 hour)
- [ ] Estimated epoch time: _______ minutes
- [ ] Expected: ~25-30 minutes
- [ ] Speedup vs single-GPU: _______×
- [ ] Expected speedup: 12-15×

### Quality validation (first few epochs)
- [ ] Epoch 0 loss: _______
- [ ] Epoch 1 loss: _______
- [ ] Epoch 2 loss: _______
- [ ] Loss trajectory matches single-GPU baseline

### Ongoing monitoring strategy
```bash
# Set up periodic checks
watch -n 600 './monitor_training.sh <JOBID>'

# Or check logs periodically
./monitor_training.sh logs stephen_voltage_ddp_20gpu
```

- [ ] Checkpoints saving every epoch
- [ ] No node failures
- [ ] No NCCL errors
- [ ] GPU utilization remains high across all nodes

---

## Troubleshooting Log

**Phase:** _______  
**Issue:** _______  
**Error message:** _______  
**Solution tried:** _______  
**Outcome:** _______  

---

## Final Results Summary

### Completed Phases
- [ ] Phase 1: Environment validation
- [ ] Phase 2: 2-GPU test
- [ ] Phase 3: 4-GPU single-node
- [ ] Phase 4: Multi-node test (optional)
- [ ] Phase 5: Full 20-GPU production

### Performance Summary
| Configuration | Epoch Time | Speedup | Loss Match |
|---------------|------------|---------|------------|
| 1 GPU (baseline) | ~6 hours | 1× | - |
| 2 GPUs | _______ | _____× | ✓/✗ |
| 4 GPUs | _______ | _____× | ✓/✗ |
| 8 GPUs | _______ | _____× | ✓/✗ |
| 20 GPUs | _______ | _____× | ✓/✗ |

### Ready for Production?
- [ ] All phases passed
- [ ] Speedup is reasonable (>10× for 20 GPUs)
- [ ] Loss convergence matches single-GPU
- [ ] No recurring errors
- [ ] Checkpoints can be loaded successfully

**Date completed:** _______  
**Notes:** _______

---

## Quick Reference Commands

```bash
# Submit jobs
sbatch train_distributed_test.sh          # 2 GPU test
sbatch train_distributed_single_node.sh   # 4 GPU
sbatch train_distributed_multi_node.sh    # 20 GPU

# Monitor
squeue -u warnet02                         # Check queue
./monitor_training.sh <JOBID>              # Monitor job
tail -f /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log  # Live logs

# Debug
scancel <JOBID>                            # Cancel job
./monitor_training.sh logs <exp_name>      # View logs
./monitor_training.sh checkpoints          # List checkpoints

# Check GPUs on node
ssh <node_name> nvidia-smi
```
