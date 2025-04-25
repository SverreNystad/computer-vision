#!/usr/bin/env bash
#SBATCH --job-name="computer-vision-snow-poles"
#SBATCH --mail-user=tim.matras@ntnu.no
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --array=1-10
#SBATCH --output=slurm_logs/slurm-job-%A_%a-%j.log   # assumes --array jobs
#SBATCH --error=slurm_logs/slurm-job-%A_%a-%j.err   # assumes --array jobs
#SBATCH --account=share-ie-idi    # ie-idi shareholder account, high priority
#SBATCH --partition=GPUQ          # the GPU queue
#SBATCH --nodes=1                 # number of nodes
#SBATCH --mem=32GB                # memory per node
#SBATCH --gres=gpu:1              # GPUs per node
#SBATCH --constraint="(p100|v100|a100|h100)&(gpu32g|gpu40g|gpu80g)"
#SBATCH --ntasks-per-node=1       # mpi processes per node, should equal number of GPUs per node
#SBATCH --cpus-per-task=4         # cpu threads per process
#SBATCH --time=3-00:00:00
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$USER/computer-vision/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.10

# Activate the environment
conda activate ${ENV_PATH}
pip install -r requirements.txt

echo "Downloaded dependencies:"
pip freeze

# Enable W&B logging for Ultralytics
yolo settings wandb=True


# Run the Python script
python train_svetimdet.py

# Deactivate the environment
conda deactivate

echo "Job finished"
