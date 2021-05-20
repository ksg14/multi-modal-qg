from tqdm import tqdm
import json
import subprocess
import os

def save_audio (save_path, video_id, video_path):
    try:
        command = f"ffmpeg -hide_banner -loglevel panic -i {video_path}/{video_id}.mp4 -ab 160k -ac 2 -ar 44100 -vn {save_path}/{video_id}.wav"
        subprocess.call(command, shell=False)
    except: 
        return 1
    return 0

if __name__ == '__main__':
    save_path = 'audio'
    video_path = 'vids'
    video_json = 'videos.json'

    if not os.path.exists (save_path):
        print (f'{save_path} dir created')
        os.mkdir (save_path)

    # ignore_ids = set ([ i for i in range (40) ])
    # ignore_ids.add (96)

    with open (video_json, 'r') as file_io:
        videos = json.load (file_io)
    
    for video in tqdm (videos):
        if len (video ['video_url']) == 0:
            break

        status = save_audio (save_path, video ['id'], video_path)
        if status == 1:
            print (f'Failed for {video ["id"]}')
            break
    print ('Done!')
