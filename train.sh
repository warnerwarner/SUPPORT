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


# Define the directory path
dir_path="/gpfs/data/shohamlab/tom/stephen/FOV10_440hz_force1s_9X_1p4x1SAM_0p59t-0p8s-FF_0p25ms-fb_bin1_EOD-on_00001_reconstructed_filtered.tif"
n_epochs=500
exp_name="stephen_voltage"
checkpoint_interval=2
training_fraction=0.5
epoch_start=11 # Don't forget the saved models are 0 indexed so if you want to start from model_n you need to set this to n+1

job_user=$(squeue -j "$SLURM_JOB_ID" -o "%u" | awk 'NR==2')
batchflag=$(squeue -j "$SLURM_JOB_ID" -O "BatchFlag" | awk 'NR==2')

# Get the total number of files in the directory
total_files=$(find "$dir_path" -type f | wc -l)

# Calculate set fraction of the total number of files
training_percent=$(awk -v var1="$total_files" -v var2="$training_fraction" 'BEGIN {print var1 * var2}')
training_percent=$(awk -v num="$training_percent" 'BEGIN{print int(num + 1)}')


mkdir -p "./splits/$exp_name"




# squeue_out=$(squeue -O "JobID,BatchFlag" -u warnet02)
# squeue_out=${squeue_out:40:-1}
# IFS=$'\n' read -d '' -ra squeue_split <<< "$squeue_out"

# # Split the squeue_split array into separate job IDs and batch flags
# job_ids=()
# batch_flags=()
# for element in "${squeue_split[@]}"; do
#     IFS=' ' read -ra split <<< "$element"
#     job_ids+=("${split[0]}")
#     batch_flags+=("${split[1]}")
# done

# # Print the job IDs and batch flags
# echo "Job IDs: ${job_ids[@]}"
# echo "Batch flags: ${batch_flags[@]}"
# echo $SLURM_JOB_ID

# for index in "${!job_ids[@]}"; do
#     if [[ "${job_ids[$index]}" == "$SLURM_JOB_ID" ]]; then
#         echo "The index of '$SLURM_JOB_ID' is: $index"
#         echo "The batch flag of '$SLURM_JOB_ID' is: ${batch_flags[$index]}"
#         batchflag=${batch_flags[$index]}
#         echo $batchflag
#         break
#     fi
# done

# if [ $batchflag == 0 ]; then
#     echo "batchflag is 0"
# else
#     echo "batchflag is not 0"
# fi

training_files=()
validating_files=()
if [ -e "./splits/$exp_name/splits.txt" ]; then
    echo "file exists"
    # File path
    input_file="./splits/$exp_name/splits.txt"

    # Using mapfile to read lines into an array
    mapfile -t my_array < "$input_file"

    # Using array elements
    for line in "${my_array[@]}"; do
        if [ "$line" == "Training files:" ]; then
            echo "found training"
            training_flag=1
            continue
        elif [ "$line" == "Validating files:" ]; then
            echo "found validating"
            training_flag=0
            continue
        fi
        if [ "$training_flag" == 1 ]; then
            training_files+=("$line")
        else
            validating_files+=("$line")
        fi
    done

fi




# Check if the file exists
if [ -e "./splits/$exp_name/splits.txt" ]; then
    echo "file exists"
    # File path
    input_file="./splits/$exp_name/splits.txt"

    # Using mapfile to read lines into an array
    mapfile -t my_array < "$input_file"

    # Using array elements
    for line in "${my_array[@]}"; do
        if [ "$line" == "Training files:" ]; then
            echo "found training"
            training_flag=1
            continue
        elif [ "$line" == "Validating files:" ]; then
            echo "found validating"
            training_flag=0
            continue
        fi
        if [ "$training_flag" == 1 ]; then
            training_files+=("$line")
        else
            validating_files+=("$line")
        fi
    done

    if [ $batchflag == 0 ]; then
        # Prompt the user for input
        read -p "File exists. Do you want to overwrite it? [y/n] " response

        # Check the user's response
        case "$response" in
            [yY][eE][sS]|[yY])
                mapfile -t files < <(find "$dir_path" -type f | shuf)
                training_files=("${files[@]:0:training_percent}")
                validating_files=("${files[@]:training_percent:total_files}")
                echo "Overwriting the file..."
                # Add your file overwriting logic here
                printf "Training files:\n" > "./splits/$exp_name/splits.txt"
                printf "%s\n" "${training_files[@]}" >> "./splits/$exp_name/splits.txt"
                printf "\n\n" >> "./splits/$exp_name/splits.txt"
                printf "Validating files:\n" >> "./splits/$exp_name/splits.txt"
                printf "%s\n" "${validating_files[@]}" >> "./splits/$exp_name/splits.txt"
                ;;
            [nN][oO]|[nN])
                echo "Not overwriting the file."
                ## TODO: Implement a way to read the previous file and use the training from that
                ;;
            *)
                echo "Invalid response."
                exit
                ;;
        esac
    else
        echo "Training files:"
        echo "${training_files[@]}"
        echo "Validating files:"
        echo "${validating_files[@]}"
    fi
else
    mapfile -t files < <(find "$dir_path" -type f | shuf)
    training_files=("${files[@]:0:training_percent}")
    validating_files=("${files[@]:training_percent:total_files}")
    printf "Training files:\n" > "./splits/$exp_name/splits.txt"
    printf "%s\n" "${training_files[@]}" >> "./splits/$exp_name/splits.txt"
    printf "\n\n" >> "./splits/$exp_name/splits.txt"
    printf "Validating files:\n" >> "./splits/$exp_name/splits.txt"
    printf "%s\n" "${validating_files[@]}" >> "./splits/$exp_name/splits.txt"
fi

python -m src.train --exp_name "$exp_name" --noisy_data "${training_files[@]}" --n_epochs "$n_epochs" --checkpoint_interval "$checkpoint_interval" --epoch "$epoch_start"
