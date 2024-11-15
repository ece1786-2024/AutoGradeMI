from openai import OpenAI
import time
import sys
import os
import pandas as pd
sys.dont_write_bytecode = True

sys.path.append(os.path.abspath("../rubric_and_sample"))
import IELTS_rubrics as rubric
client = OpenAI()


generator = """
You are an IELTS writing section examiner. Given the following writing question and student essay, provide comments and feedbacks for this student to improve on the essay.
The feedback must be constructive, and provide an additional section for overall suggestions and improvements and an overall summary. Here is an IELTS Rubric for your reference.\n
"""
# For now, just use the basic rubric
generator += f"Rubric: {rubric.BASIC_RUBRIC}\n\n"

generator += """Also, there will be an evaluation for the generated feedback. Please revise the feedback according to the evaluation if the feedback is not good enough.
Below is the writing question and the student essay for feedback:
"""


evaluator = """
You should evaluate the generated feedback for the IELTS writing response. 
Evaluate the quality of the feedback based on authenticity and effectiveness. 
Specifically, consider whether the feedback is constructive, 
addresses key aspects of writing (such as grammar, coherence, vocabulary, and task completion).
If the feedback is good enough, just say "good" If not, give some revision suggestions. 
Below is the writing question and the student essay for reference:
"""






def get_response(role_prompt, conversation_history):
    if role_prompt == "generator":
        messages = [
            {"role": "system", "content": generator},
            {"role": "user", "content": f"Based on this conversation, imporve the generated feedback based on the evaluation.\n\n{conversation_history}"}
        ]
    else:
        messages = [
            {"role": "system", "content": evaluator},
            {"role": "user", "content": f"Based on this conversation, give an evaluation on the generated feedback.\n\n{conversation_history}"}
        ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content



if __name__ == "__main__":
    # file path just used for testing
    val_path = "../../data/dataset/processed/validation.csv"
    df =  pd.read_csv(val_path, encoding='utf-8')
    df = df.iloc[[0]]
    prompt_list = df["prompt"]
    essay_list = df["essay"]
    prompt_example = prompt_list[0]
    essay_example = essay_list[0]
    generator += f"\nPrompt: {prompt_example}\n\nEssay: {essay_example}"
    evaluator += f"\nPrompt: {prompt_example}\n\nEssay: {essay_example}"

    # Let the generator start the conversation by providing a feedback
    conversation = []
    messages = [
        {"role": "system", "content": generator},
        {"role": "user", "content": f"Give the feedback please."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    first_response = response.choices[0].message.content
    conversation.append(f"generator: {first_response}")
    




    # Just 3 turns for now
    for i in range(3):
        current_conversation = "\n".join(conversation)
        
        evaluator_response = get_response(evaluator, current_conversation)
        conversation.append(f"evaluator: {evaluator_response}")
        time.sleep(0.5)
        
        # Update conversation for generator's turn
        current_conversation = "\n".join(conversation)
        
        # Generator's turn
        generator_response = get_response(generator, current_conversation)
        conversation.append(f"generator: {generator_response}")
        time.sleep(0.5)
        
        print(f"Turn {i+1} finished")

        
    with open("conversations_log.txt", "w", encoding='utf-8') as f:
        for line in conversation:
            f.write(line + "\n")

    print("Finish")