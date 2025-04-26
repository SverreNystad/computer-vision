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

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.6.0

source .venv/bin/activate

python train_svetimdet.py

