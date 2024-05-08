#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-prep-%j.log
sync;sync;sync

echo `hostname`s

source ~/.bashrc

conda deactivate
conda activate glofe

export PYTHONPATH=.

for i in {0..2};
do
    echo "job $i"
    sh util_job.sh "$i"
done