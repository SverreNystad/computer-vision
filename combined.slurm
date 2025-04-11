#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=combined-optuna-search
#SBATCH --time=3-00:25:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=4
#SBATCH --constraint="(p100|v100|a100|h100)&(gpu16g|gpu32|gpu40|gpu80g)"
#SBATCH --output=output_combined.txt
#SBATCH --error=output_combined.err
#SBATCH --mail-user=sverrnys@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.10

# Activate the environment
conda activate ${ENV_PATH}
cd computer-vision/
pip install -r requirements.txt

echo "Downloaded dependencies:"
pip freeze

# Run the Python script
python train-combined.py

# Deactivate the environment
conda deactivate

echo "Job finished"


# "sbatch ./combined.slurm"  to run the job