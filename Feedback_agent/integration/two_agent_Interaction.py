from openai import OpenAI
import time

client = OpenAI()


generator = """
You are an IELTS writing section examiner. 
Given the following writing question and student essay, provide comments and feedbacks for this student to improve on the essay.
There will be an evaluation for the feedback provided. Please revise the feedback according to the evaluation if the feedback is not good enough.
"""

evaluator = """
You should evaluate the generated feedback for the IELTS writing response. 
Evaluate the quality of the feedback based on authenticity and effectiveness. 
Specifically, consider whether the feedback is constructive, 
addresses key aspects of writing (such as grammar, coherence, vocabulary, and task completion).
If the feedback is good enough, just say "good" If not, give some revision suggestions. 
"""





def get_response(role_prompt, conversation_history):
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": f"Based on this conversation, what would you say next?\n\n{conversation_history}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


# Unfinished here, Initialize conversation to let the generator make a feedback first
conversation = [""]


# Just 3 turns for now
for i in range(3):
    current_conversation = "\n".join(conversation)
    
    evaluator_response = get_response(evaluator, current_conversation)
    conversation.append(f"evaluator: {evaluator_response}")
    time.sleep(1)
    
    # Update conversation for generator's turn
    current_conversation = "\n".join(conversation)
    
    # Generator's turn
    generator_response = get_response(generator, current_conversation)
    conversation.append(f"generator: {generator_response}")
    time.sleep(1)
    
    print(f"Turn {i+1} finished")

    
with open("conversations_log.txt", "w", encoding='utf-8') as f:
    for line in conversation:
        f.write(line + "\n")

print("Finish")