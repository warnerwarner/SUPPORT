# Distributed Training Implementation for SUPPORT

This directory contains distributed PyTorch training scripts using DistributedDataParallel (DDP) with NCCL backend.

## Files Created

### Training Scripts
- **`src/train_distributed.py`** - Main distributed training script (DDP implementation)
- **`src/train.py`** - Original single-GPU training script (unchanged)

### SLURM Job Scripts
- **`train_distributed_test.sh`** - 2 GPU test (1 node × 2 GPUs, 2 epochs)
- **`train_distributed_single_node.sh`** - Single-node training (1 node × 4 GPUs, 500 epochs)
- **`train_distributed_multi_node.sh`** - Multi-node training (5 nodes × 4 GPUs = 20 GPUs, 500 epochs)

## Testing Strategy

### Phase 1: Syntax Validation (COMPLETED ✓)
```bash
# Already verified:
# - Python syntax check passed
# - SLURM scripts syntax check passed
# - Scripts are executable
```

### Phase 2: 2-GPU Test (RECOMMENDED FIRST STEP)
```bash
cd /gpfs/data/shohamlab/tom/support
sbatch train_distributed_test.sh

# Monitor the job
squeue -u warnet02

# Watch logs in real-time
tail -f /gpfs/home/warnet02/shohamlab/tom/tmp/<JOBID>.log
```

**What to check:**
- World size = 2
- Both GPUs show high utilization
- Loss decreases over batches
- Checkpoints saved in `results/saved_models/ddp_2gpu_test/`
- No NCCL errors

**Expected time:** ~30 minutes for 2 epochs

---

### Phase 3: 4-GPU Single-Node Test
```bash
sbatch train_distributed_single_node.sh

# Monitor GPU usage on the compute node
# (get node name from squeue output)
ssh <node_name>
watch -n 1 nvidia-smi
```

**Expected speedup:** 3-3.5× vs single GPU (epoch time ~1.7-2 hours)

---

### Phase 4: Full Production (20 GPUs)
**Only proceed if Phase 3 succeeded!**

```bash
sbatch train_distributed_multi_node.sh
```

**Expected speedup:** 12-15× vs single GPU (epoch time ~25-30 minutes)

---

## Key Features

### Distributed Training Implementation
- **NCCL backend** for fast GPU-GPU communication
- **DistributedSampler** ensures no data duplication across GPUs
- **Gradient synchronization** via DDP wrapper
- **Rank 0 only** handles logging and checkpointing
- **Automatic batch distribution** across all GPUs

### Performance Optimizations
- **AMP (Automatic Mixed Precision)** enabled by default
- **Hierarchical all-reduce** (intra-node first, then inter-node)
- **Prefetching** and parallel data loading
- **Pin memory** for faster CPU→GPU transfer

---

## Monitoring Commands

### Check job status
```bash
squeue -u warnet02 -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R"
```

### Watch training logs
```bash
tail -f results/logs/<exp_name>.log | grep "loss :"
```

### Monitor GPU utilization
```bash
# On compute node
ssh <node_name>
nvidia-smi dmon -s u
```

### Check checkpoints
```bash
ls -lh results/saved_models/<exp_name>/
```

---

## Troubleshooting

### Job hangs at initialization
**Symptom:** Stuck at "Initializing distributed training"

**Solutions:**
1. Check MASTER_ADDR is reachable: `ping $MASTER_ADDR`
2. Verify NCCL available: already confirmed ✓
3. Increase timeout (already set in code)

### CUDA out of memory
**Solutions:**
1. Reduce batch size: `--batch_size 8`
2. Reduce workers: `--n_cpu 4`
3. Check GPU memory before job: `nvidia-smi`

### Different loss across ranks
**Cause:** DDP not synchronizing gradients properly

**Check:** Model type should be `DistributedDataParallel`

### Permission errors
**Solution:** Already handled with `dist.barrier()` to sync directory creation

---

## Configuration

### Current Settings (train_distributed_single_node.sh)
```bash
Partition: gpu4_medium
Nodes: 1
GPUs per node: 4
Total GPUs: 4
Batch size per GPU: 16
Effective batch size: 64
Epochs: 500
Checkpoint interval: Every epoch
```

### To Modify Parameters
Edit the SLURM script variables:
```bash
n_epochs=500           # Number of training epochs
exp_name="..."         # Experiment name for logs/checkpoints
checkpoint_interval=1  # Save every N epochs
```

Or pass additional arguments to the Python script:
```bash
--batch_size 16        # Batch size per GPU
--patch_interval 10 64 64  # Patch sampling interval
--lr 5e-4             # Learning rate
--depth 5             # Model depth
--bs_size 3 3         # Blind spot size
```

---

## Expected Results

### Performance Comparison
| Configuration | GPUs | Speedup | Epoch Time | Total (500 epochs) |
|---------------|------|---------|------------|-------------------|
| Original (1 GPU) | 1 | 1× | ~6 hours | ~125 days |
| Single-node | 4 | 3.5× | ~1.7 hours | ~36 days |
| Multi-node | 20 | 13× | ~28 min | ~10 days |

### Convergence
- Loss trajectory should match single-GPU training
- Effective batch size = batch_size × world_size
- Learning rate is kept at 5e-4 (may adjust if convergence differs)

---

## Next Steps

1. ✅ **Implementation complete** - All files created and syntax verified
2. **Run 2-GPU test** - Validate distributed training works
3. **Run 4-GPU single-node** - Validate performance and scaling
4. **Run 20-GPU multi-node** - Production training

---

## Contact

For issues or questions about this implementation, contact Tom Warner (tom.warner@nyulangone.org)

Last updated: 2025-02-02
