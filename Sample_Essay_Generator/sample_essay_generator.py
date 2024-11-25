import os
import sys
import pandas as pd
from openai import OpenAI
import numpy as np

sys.path.append(os.path.abspath(".."))
from Feedback_agent.rubric_and_sample import IELTS_rubrics as rubric

def get_score_prompt_version(topic, essay):
    client = OpenAI()
    grader_prompt = f"""
    You are an IELTS writing section examiner. 
    Given the writing queston and the student essay, please grade the essay on a scale of 0 to 9 based on the IELTS Rubric and 0.5 intervals are allowed.

    Writing Question: {topic}
    Student Essay: {essay}

    Here is an IELTS rubric for your reference: 
    Rubric: 
    {rubric.BASIC_RUBRIC}
    {rubric.CRITERIA}
    {rubric.BAND_SCORE}

    Please output the score of the essay in the form of 'score of the essay'. Please output the score directly.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": grader_prompt}
        ]
    )
    return response.choices[0].message.content



def generate_sample_essay(topic, essay, feedback, predicted_grade, desired_grade):
    client = OpenAI()
    prompt = f"""
    You are an IELTS writing assistant for IELTS test takers. 
    Given the writing question with student essay, feedback, the predicted score of the essay, and student's desired score. Please generate a sample essay of the desired score based on the provided writing question, student essay, feedback, predicted score, and desired score. 

    Writing Question: {topic}
    Student Essay: {essay}
    Feedback: {feedback}
    Predicted Score: {predicted_grade}
    Desired Score: {desired_grade}

    Here is an IELTS rubric for your reference: 
    Rubric: 
    {rubric.BASIC_RUBRIC}
    {rubric.CRITERIA}
    {rubric.BAND_SCORE}

    Please consider the feedback carefully and the sample essay should cover the arguments introduced in the student's essay.
    Please ONLY output the sample essay. 
    """
    
    #print("Generating essay ...... \n")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def check_score_input(score):
    score = float(score)
    print(score)
    if score not in list(np.arange(0.0, 9.5, 0.5)):
        return False
    return True

def main(mode="auto"):
    if mode == 'auto':
        feedback_path = "../Feedback_agent/Generator_merged/feedback.csv"
        df = pd.read_csv(feedback_path, encoding='utf-8')

        topic = df["topic"]
        essay = df["essay"]
        feedback = df["feedback"]
        predicted_grade = df["predicted"]
        desired_grade = df["desired"]
    else:
        topic = input("Enter the writing question/topic: ")
        essay  = input("Enter the student's essay: ")
        feedback = input("Enter the feedback from generator: ")
        predicted_grade = input("Enter the predicted grade: ")
        desired_grade = input("Enter your desired grade: ")

        topic = list(topic)
        essay = list(essay)
        feedback = list(feedback)
        predicted_grade = list(predicted_grade)
        desired_grade = list(desired_grade)

    # input check
    sample_score_list = []
    sample_essay_list = []
    print("generating essays ..... \n")
    for (q, e, f, ps, ds) in zip(topic, essay, feedback, predicted_grade, desired_grade):
        pred_status = check_score_input(ps)
        desired_grade_status = check_score_input(ds)
        if(pred_status is False or desired_grade_status is False):
            print("The given score does not follow IELTS grading band score standards\n")
            return None
        if(ps > ds):
            print("predicted score is already higher that desired grade, abort!\n")
            return None

        sample_score = 0.0
        sample_essay = ""
        appended = False
        i = 0
        while i < 3:
            # double check this loop, seems a bit dangerous
            sample_essay = generate_sample_essay(q, e, f, ps, ds)
            sample_score = get_score_prompt_version(q, sample_essay)
            if float(sample_score) >= float(ds):
                sample_essay_list.append(sample_essay)
                sample_score_list.append(sample_score)
                appended = True
                break
            i += 3

        if appended is False:
            # just append the latest
            sample_essay_list.append(sample_essay)
            sample_score_list.append(sample_score)

    # export result
    # Export feedback to csv
    export_dict = dict()
    export_dict["topic"] = topic
    export_dict["predicted"] = predicted_grade
    export_dict["desired"] = desired_grade
    export_dict["feedback"] = feedback
    export_dict["essay"] = essay
    export_dict["sample_essay"] = sample_essay_list
    export_dict["sample_score"] = sample_score_list
    

    df = pd.DataFrame(export_dict)
    df.to_csv("./sample_essay.csv", index=False, encoding='utf-8', mode='w')
    print("DONE!!!")
    return None


if __name__ == "__main__":
    # two mode: auto or other
    # other means user input option
    main(mode="auto")
