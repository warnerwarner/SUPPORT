#!/bin/bash

# Define the variations you want to test
BS_SIZES=("1 3" "2 2" "3 3")
PHASE_FLAGS=("--use_phase_conditioning")

echo "Starting sweep submission..."
echo "============================="

for bs in "${BS_SIZES[@]}"; do
    for phase in "${PHASE_FLAGS[@]}"; do
        
        # Format the bs_size for the experiment name (e.g., "1_1")
        bs_name=${bs/ /_}
        
        # Determine if phase is used for the name
        if [ -z "$phase" ]; then
            phase_name="no_phase"
        else
            phase_name="with_phase"
        fi
        
        # Create a unique experiment name
        EXP_NAME="stephenvoltage_bs_l105l205_${bs_name}_${phase_name}_full_phase"
        
        echo "Launching job: $EXP_NAME"
        echo "  - BS Size: $bs"
        echo "  - Phase: $(if [ -z "$phase" ]; then echo "None"; else echo "Enabled"; fi)"
        
        # Launch the job, exporting our variables to the sbatch environment
        sbatch --export=ALL,BS_SIZE="$bs",PHASE_FLAG="$phase",EXP_NAME="$EXP_NAME" train_sweep_job.sh
        
        # Brief pause to not overwhelm the scheduler
        sleep 1
    done
done

echo "============================="
echo "All jobs submitted!"
