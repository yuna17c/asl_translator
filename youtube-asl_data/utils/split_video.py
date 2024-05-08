import os
import json
import math
import cv2
import sys
import glob
import subprocess
import shutil
import tempfile
import argparse
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import glob


def get_clip(input_video_dir, output_video_dir, tsv_fn, rank, nshard, target_size=224, ffmpeg=None):
    os.makedirs(output_video_dir, exist_ok=True)
    df = pd.read_csv(tsv_fn, sep='\t')
    mp4_files = glob.glob(input_video_dir+'/*.mp4')

    mp4_downloaded = {}
    for path in mp4_files:
        yid = path.split('-')[-1].split('.mp4')[0]
        mp4_downloaded[yid] = path
    
    items = []
    for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
        if yid not in mp4_downloaded.keys():
            continue
        items.append([vid, yid, start, end])
    
    num_per_shard = (len(items)+nshard-1)//nshard
    items = items[num_per_shard*rank: num_per_shard*(rank+1)]
    print(f"{len(items)} videos")
    for vid, yid, start_time, end_time in tqdm(items):
        output_video = os.path.join(output_video_dir, vid+'.mp4')
        input_video_whole = mp4_downloaded[yid]
        if os.path.isfile(output_video):
            continue
        cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-crf', '20', output_video]
        print(' '.join(cmd))
        subprocess.call(cmd)
    return

def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='data tsv file')
    parser.add_argument('--raw', type=str, help='raw video dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='path to ffmpeg')
    parser.add_argument('--target-size', type=int, default=224, help='image size')

    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    args = parser.parse_args()

    get_clip(args.raw, args.output, args.tsv, 0, 1, target_size=args.target_size, ffmpeg=args.ffmpeg)
    return


if __name__ == '__main__':
    main()

