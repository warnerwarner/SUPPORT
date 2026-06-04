#!/bin/bash
# Helper script to monitor distributed training jobs

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SUPPORT Distributed Training Monitor"
echo "=========================================="
echo ""

# Check for running jobs
echo -e "${GREEN}Your Running Jobs:${NC}"
squeue -u warnet02 -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R"
echo ""

# Get job ID if provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    echo ""
    echo "Commands:"
    echo "  ./monitor_training.sh <job_id>     # Monitor specific job"
    echo "  ./monitor_training.sh logs <exp>   # View training logs"
    echo "  ./monitor_training.sh checkpoints  # List checkpoints"
    echo ""
    exit 0
fi

JOB_ID=$1

if [ "$JOB_ID" == "logs" ]; then
    EXP_NAME=${2:-"ddp_2gpu_test"}
    echo -e "${GREEN}Training logs for $EXP_NAME:${NC}"
    if [ -f "results/logs/$EXP_NAME.log" ]; then
        tail -50 results/logs/$EXP_NAME.log | grep -E "(loss|Epoch|ERROR)"
    else
        echo -e "${RED}Log file not found: results/logs/$EXP_NAME.log${NC}"
    fi
    exit 0
fi

if [ "$JOB_ID" == "checkpoints" ]; then
    echo -e "${GREEN}Available checkpoints:${NC}"
    ls -lh results/saved_models/*/model_*.pth 2>/dev/null | tail -20
    exit 0
fi

echo -e "${GREEN}Job Details:${NC}"
scontrol show job $JOB_ID | grep -E "(JobId|JobName|Partition|NodeList|State|StartTime|RunTime)"
echo ""

# Get nodes
NODES=$(squeue -j $JOB_ID -h -o "%N")
if [ -z "$NODES" ]; then
    echo -e "${RED}Job $JOB_ID not found or not running${NC}"
    exit 1
fi

echo -e "${GREEN}Nodes: ${NC}$NODES"
echo ""

# Get log file
LOG_FILE="/gpfs/home/warnet02/shohamlab/tom/tmp/${JOB_ID}.log"

if [ -f "$LOG_FILE" ]; then
    echo -e "${GREEN}Last 30 lines of log:${NC}"
    tail -30 "$LOG_FILE"
    echo ""
    echo -e "${YELLOW}To follow logs in real-time:${NC}"
    echo "  tail -f $LOG_FILE"
else
    echo -e "${YELLOW}Log file not yet created: $LOG_FILE${NC}"
fi

echo ""
echo -e "${GREEN}GPU Status on compute nodes:${NC}"
# Expand node range if needed
for node in $(scontrol show hostname $NODES); do
    echo "--- $node ---"
    ssh $node "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || echo "  (unable to connect)"
done

echo ""
echo -e "${YELLOW}Commands:${NC}"
echo "  Watch this job: watch -n 5 './monitor_training.sh $JOB_ID'"
echo "  Cancel job:     scancel $JOB_ID"
echo "  View full log:  less $LOG_FILE"
