import os
import yt_dlp as youtube_dl
from yt_dlp import YoutubeDL
from tqdm import tqdm
import pandas as pd
import re
import subprocess
from bbox import bbox, test_bbox
import json
import sys

def download_data(link_file, output, loop_n, num = 100):
    start = int(loop_n) * 100
    print('start:', loop_n)
    with open(link_file, 'r') as file:
        for _ in range(start):
            next(file)

        url_endings = [next(file).strip() for _ in range(num)]
        urls = ['https://www.youtube.com/watch?v=' + url_end for url_end in url_endings]

    # download options
    ydl_opts = {
        'format': 'best', 
        'outtmpl': output + '%(title)s-%(id)s.%(ext)s',
        'ignoreerrors': True,
        'writesubtitles': True, 
        'subtitlesformat': 'srt',  
        'subtitleslangs': ['en'],  
    }
    # download yt
    with YoutubeDL(ydl_opts) as ydl:
        for url in tqdm(urls, desc="Downloading videos and subtitles", total=num):
            ydl.download([url])


def parse_vtt_to_dataframe(vtt_file_path):
    youtube_id = vtt_file_path.split('-')[-1].split('.en.vtt')[0]
    def tokenize_text(text):
        return re.sub(r'[^\w\s]', '', text.lower())

    #read vtt
    with open(vtt_file_path, 'r', encoding='utf-8') as file:
        vtt_lines = file.readlines()

    # process
    data = []
    for line in vtt_lines:
        if '-->' in line:
            start, end = line.strip().split(' --> ')
            caption_index = vtt_lines.index(line) + 1
            if caption_index < len(vtt_lines):
                raw_text = vtt_lines[caption_index].strip()
                tokenized_text = tokenize_text(raw_text)
                vid = '-'.join([youtube_id, start,end])
                data.append([vid, youtube_id, start, end, raw_text, tokenized_text, None, 'test'])

    # conv to df
    df = pd.DataFrame(data, columns=['vid', 'yid', 'start', 'end', 'raw-text', 'tokenized-text', 'gloss', 'split'])
    output_tsv_path = vtt_file_path.replace('data/', 'data/tsv_files/')[:-3] + 'tsv'
    print(f'string: {output_tsv_path}')
    # df.to_csv(output_tsv_path, sep='\t', index=False)
    return df


def all_vtt_process(folder_path):
    vtt_files = [f for f in os.listdir(folder_path) if f.endswith('.vtt')]
    total_df = pd.DataFrame()
    for vtt_file in tqdm(vtt_files):
        vtt_file_path = os.path.join(folder_path, vtt_file)
        print(vtt_file_path)
        df = parse_vtt_to_dataframe(vtt_file_path)
        total_df = pd.concat([total_df,df])

    output_tsv_path = 'data/tsv_files/youtube-asl_v1.tsv'
    total_df.to_csv(output_tsv_path, sep='\t', index=False)


def run_bbox(folder_path):
    vids = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    bounding_boxes = {}
    for vid in tqdm(vids):
        url = vid.split('.mp4')[0]
        box = bbox(folder_path, vid)
        bounding_boxes[url] = box

    # Write the bounding box coordinates to a JSON file
    with open("utils/bounding_boxes.json", "w") as json_file:
        json.dump(bounding_boxes, json_file, indent=4)
        json_file.write('\n')

    print("Bounding box coordinates saved to bounding_boxes.json")


def split_video(tsv_path, input_path, output_path):
    split_command = [
        'python', 
        'utils/split_video.py', 
        '--tsv', tsv_path, 
        '--raw', input_path, 
        '--output', output_path, 
        '--ffmpeg', 'ffmpeg'
    ]
    subprocess.run(split_command)

def run_crop_resize():
    cmd = [
        'python', 
        'utils/crop_resize.py',
        '--bbox', 'utils/bounding_boxes.json',
        '--raw', 'utils/video-clip', 
        '--output', 'utils/split-resized-vids', 
        '--ffmpeg', 'ffmpeg'
    ]
    subprocess.run(cmd)

def write_to_txt(vid_path):
    vids = [f for f in os.listdir(vid_path) if f.endswith('.mp4')]
    root_dir = '/home/grt/youtube-asl_data'
    txt = ''
    for vid in vids:
        path = os.path.join(root_dir, vid_path, vid)
        txt += path + '\n'
    # print(txt)
    with open('../GloFE/tools/youtube_asl_samples.txt', 'w') as file:
        file.write(txt)

def num_videos():
    vids = [f for f in os.listdir('data') if f.endswith('.mp4')]
    splits = [f for f in os.listdir('utils/video-clip') if f.endswith('.mp4')]
    print("total number of videos: ", len(vids))
    print("total number of split videos: ", len(splits))


arg = sys.argv[1]
print(arg)
download_data('youtube-asl_youtube_asl_video_ids.txt', 'data/', arg, num = 100)
all_vtt_process('data')
split_video('data/tsv_files/youtube-asl_v1.tsv', 'data','utils/video-clip')
run_bbox('utils/video-clip')
# test_bbox('utils/video-clip','utils/bbox-demo')
run_crop_resize()
write_to_txt('utils/split-resized-vids')
num_vids = num_videos()