from tqdm import tqdm
import json
import subprocess
import os

def clip_audio (save_path, video_id, question_id, answer_start, answer_end, audio_path):
    try:
        if len (answer_start) == 4:
            answer_start = '0' + answer_start
        if len (answer_end) == 4:
            answer_end = '0' + answer_end

        command = f"ffmpeg -hide_banner -loglevel panic -i {audio_path}/{video_id}.wav -ss 00:{answer_start} -to 00:{answer_end} -c copy {save_path}/v_{video_id}_q_{question_id}_.wav"
        subprocess.call(command, shell=False)
    except: 
        return 1
    return 0

if __name__ == '__main__':
    save_path = 'salient_audio_clip'
    audio_path = 'audio'
    questions_json = 'labelled_questions.json'

    if not os.path.exists (save_path):
        print (f'{save_path} dir created')
        os.mkdir (save_path)

    # ignore_ids = set ([ i for i in range (40) ])
    # ignore_ids.add (96)

    with open (questions_json, 'r') as file_io:
        questions = json.load (file_io)

    id_to_use = set ([0, 67])
    
    for question in tqdm (questions):
        if len (question ['question']) == 0:
            break

        if question ['question_id'] not in id_to_use:
            continue

        status = clip_audio (save_path, question ['video_id'], question ['question_id'], question ['answer_start'], question ['answer_end'], audio_path)
        if status == 1:
            print (f"Failed for {question ['question_id']}")
            break
    print ('Done!')
