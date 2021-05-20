import json

def assign_question_ids (start_id, questions):
    for question in questions:
        question ['question_id'] = start_id

        start_id += 1
    return questions

if __name__ == '__main__':
    questions_path = 'questions.json'
    labelled_questions_path = 'labelled_questions.json'

    with open (questions_path, 'r') as file_io:
        questions = json.load (file_io)

    labelled_questions = assign_question_ids (start_id=0, questions=questions)

    with open (labelled_questions_path, 'w') as file_io:
        json.dump (labelled_questions, file_io)
    
    print ('Done!')
