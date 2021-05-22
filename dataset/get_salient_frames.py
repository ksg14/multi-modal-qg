from tqdm import tqdm
import json
import os
import cv2 
import math
import numpy as np

def is_salient (frame_time, start_time, end_time):
    if frame_time >= start_time and frame_time <= end_time:
        return True
    return False    

def save_salient_frames (save_path, frame_dim, video_id, question_id, answer_start, answer_end, video_path):
    start_m, start_s = answer_start.split (':')
    end_m, end_s = answer_end.split (':')

    start_time = (int (start_m)*60 + int (start_s)) * 1000 # milliseconds
    end_time = (int (end_m)*60 + int (end_s)) * 1000 # milliseconds
    
    frame_time_param = 0
    frame_rate_param = 5
    frame_id_param = 1

    salient_frames = list ()
    try:
        vidObj = cv2.VideoCapture(f'{video_path}/{video_id}.mp4') 

        frame_rate = math.ceil (vidObj.get (frame_rate_param))
        
        while (vidObj.isOpened ()): 
            frame_id = vidObj.get (frame_id_param)
            frame_time = vidObj.get (frame_time_param)
    
            success, image = vidObj.read()

            if success != True:
                break
            
            if frame_id % frame_rate == 0:
                if is_salient (frame_time, start_time, end_time):
                    resized_img = cv2.resize(image, frame_dim, interpolation = cv2.INTER_AREA)

                    salient_frames.append (resized_img.tolist ())
        
        salient_frames = np.array (salient_frames)
        np.save (f'{save_path}/v_{video_id}_q_{question_id}_.npy', salient_frames)

    except Exception as e:
        print (f'Error - {str (e)}')
        return 1
    return 0

if __name__ == '__main__':
    save_path = 'salient_frames'
    video_path = 'vids'
    questions_json = 'labelled_questions.json'
    frame_dim = (112, 112)

    if not os.path.exists (save_path):
        print (f'{save_path} dir created')
        os.mkdir (save_path)

    with open (questions_json, 'r') as file_io:
        questions = json.load (file_io)
    
    for question in tqdm (questions):
        if len (question ['question']) == 0:
            break

        status = save_salient_frames (save_path, frame_dim, question ['video_id'], question ['question_id'], question ['answer_start'], question ['answer_end'], video_path)

        if status == 1:
            print (f"Failed for {question ['question_id']}")
            break

    print ('Done!')
