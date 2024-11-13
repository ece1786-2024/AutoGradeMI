# Feedback generator main -- use GPT-4o
import prompt_feedback_generation as p
from openai import OpenAI
import pandas as pd 

def main(mode="normal"):
    mode_list = ["normal", "debug_1", "debug_5"]
    if mode not in mode_list:
        print("Invalid mode passed")
        return None

    # there is no training and validation involved
    # I simply want to use the two files, and the name is only to make them distinct
    train_path = "../data/dataset/processed/train.csv"
    val_path = "../data/dataset/processed/validation.csv"

    # for now just use validation 
    df = pd.read_csv(val_path, encoding='utf-8')
    if(mode == "debug_1"):
        df = df.iloc[[0]]
    elif(mode == "debug_5"):
        df = df[1:6]

    question_list = df["prompt"]
    essay_list = df["essay"]

    feedback_list = []
    client = OpenAI()
    for (q, e) in zip(question_list, essay_list):
        user_content = p.create_user_prompt(q,e)
        system_content = p.create_system_prompt()
        completion = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system", 
                    "content": system_content
                }, 
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        res = completion.choices[0].message.content
        feedback_list.append(res)
    
    # prep for the evaluator and human evaluator
    # for human evaluators, the program exports
    feedback_df = pd.DataFrame({"feedback": feedback_list})
    df['feedback'] = feedback_df['feedback']

    # export file current directory
    df.to_csv('./feedback.csv', index=False, encoding='utf-8', mode="w")
    return None


if __name__ == "__main__":
    main("debug_1")