#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-prep-%j.log
# python utils/util.py $1
sh /home/grt/GloFE/tools/extract.sh 0 1 0

# rewrite with new split
python rewrite_tsv.py $1

# remove the videos before the next batch 
# rm -r /home/grt/youtube-asl_data/utils/split-resized-vids/*
# rm -r /home/grt/youtube-asl_data/utils/video-clip/*
find /home/grt/youtube-asl_data/data -type f -delete
