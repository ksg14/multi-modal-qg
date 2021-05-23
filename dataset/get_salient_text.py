from typing_extensions import final
from tqdm import tqdm
import json
import os

import webvtt
import pysrt

def is_important (answer_start, answer_end, start, end):
    if answer_start <= start <= answer_end:
        return True
    if answer_start <= end <= answer_end:
        return True
    if start <= answer_start <= end:
        return True
    return False

def preprocess_text (subtitle):
    lower_case = subtitle.lower ()
    final_subtitle = lower_case.replace ('\n', ' ')
    final_subtitle = final_subtitle.strip ()
    return final_subtitle

def get_vtt_context (sub_file, answer_start, answer_end):
    sentences = []

    for caption in webvtt.read(sub_file):
        start = int (caption.start [3:5])*60 + int (caption.start [6:8])
        end = int (caption.start [3:5])*60 + int (caption.start [6:8])

        subtitle = preprocess_text (caption.text)

        if len (subtitle) > 0:
            if is_important (answer_start, answer_end, start, end):
                sentences.append (subtitle)
    
    if len (sentences) == 0:
        return None
    return ' '.join (sentences)

def get_srt_context (sub_file, answer_start, answer_end):
    sentences = []

    for caption in pysrt.open (sub_file):
        start = caption.start.minutes*60 + caption.start.seconds
        end = caption.end.minutes*60 + caption.end.seconds

        subtitle = preprocess_text (caption.text)

        if len (subtitle) > 0:
            if is_important (answer_start, answer_end, start, end):
                sentences.append (subtitle)
    
    if len (sentences) == 0:
        return None
    return ' '.join (sentences)

def get_salient_text (question, subs_path):

    if os.path.exists (f"{subs_path}/{question ['video_id']}.srt"):
        extension = 'srt'
        sub_file = f"{subs_path}/{question ['video_id']}.srt"
    elif os.path.exists (f"{subs_path}/{question ['video_id']}.vtt"):
        extension = 'vtt'
        sub_file = f"{subs_path}/{question ['video_id']}.vtt"
    else:
        return None

    salient_text = dict()

    salient_text ['question_id'] = question ['question_id']
    salient_text ['video_id'] = question ['video_id']
    salient_text ['question'] = preprocess_text (question ['question'])

    start_m, start_s = question ['answer_start'].split (':')
    end_m, end_s = question ['answer_end'].split (':')
    start_time = (int (start_m)*60 + int (start_s))
    end_time = (int (end_m)*60 + int (end_s))
    
    if extension == 'srt':
        salient_text ['context'] = get_srt_context (sub_file, start_time, end_time)
    else:
        salient_text ['context'] = get_vtt_context (sub_file, start_time, end_time)
    
    if salient_text ['context'] == None:
        return None
    
    salient_text ['answer'] = preprocess_text (question ['option_1'])

    return salient_text

if __name__ == '__main__':
    save_path = 'salient_text'
    output_file_name = 'salient_text_list.json'
    subs_path = 'subs'
    questions_json = 'labelled_questions.json'
    if not os.path.exists (save_path):
        print (f'{save_path} dir created')
        os.mkdir (save_path)

    with open (questions_json, 'r') as file_io:
        questions = json.load (file_io)
    
    salient_text_list = list ()
    for question in tqdm (questions):
        if len (question ['question']) == 0:
            break

        salient_obj = get_salient_text (question, subs_path)
        if salient_obj:
            salient_text_list.append (salient_obj)

    with open (f'{save_path}/{output_file_name}', 'w') as file_io:
        json.dump (salient_text_list, file_io)

    print ('Done!')
