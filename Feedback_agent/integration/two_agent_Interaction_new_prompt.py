from openai import OpenAI
import time

client = OpenAI()


generator_prompt = """
You are an IELTS writing section examiner. 
Given the following writing question and student essay, provide comments and feedbacks for this student to improve on the essay.
Writing Question: {topic}
Student Essay: {essay}
There will be an evaluation for the feedback provided. Please revise the feedback according to the evaluation if the feedback is not good enough.
"""

evaluator_prompt = """
You should evaluate the generated feedback for the IELTS writing response. 
Writing Question: {topic}
Student Essay: {essay}
Evaluate the provided feedback on the IELTS essay according to these criteria: 
Clarity: Ensure the feedback is easy to understand and avoids ambiguous language. Good feedback should use simple and precise wording, so the essay writer knows exactly what is meant.
Relevance: Check if the feedback addresses the main components of IELTS scoring (Task Achievement, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy). Good feedback covers each of these areas as relevant to the essay.
Specificity: Look for details in the feedback. Comments like 'Improve grammar' are not specific enough. Good feedback should mention exact issues (e.g., 'There are multiple subject-verb agreement errors') and, if possible, give examples from the essay.
Actionable Suggestions: Verify that the feedback includes clear, constructive advice that the writer can apply to improve. Instead of saying 'The vocabulary is weak,' a good suggestion might be, 'Try to use more varied synonyms for common words like "important" (e.g., "crucial," "vital") to improve Lexical Resource.'
Tone: Check if the feedback is encouraging and respectful. A supportive tone can help motivate the writer to make improvements.
Continue to review the feedback until all criteria are met. If the feedback meets these criteria, respond with 'good' If not, identify the areas needing improvement."
"""


def get_response(role_prompt, topic, essay, conversation_history):
    prompt = role_prompt.format(topic=topic, essay=essay)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Based on this conversation, what would you say next?\n\n{conversation_history}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


def generate_feedback(topic, essay):
    conversation = []
    feedback_good = False
    good_feedback = None

    for turn in range(10):  # Maximum limit to avoid infinite loop
        current_conversation = "\n".join(conversation)

        # Generator's turn
        generator_response = get_response(generator_prompt, topic, essay, current_conversation)
        conversation.append(f"generator: {generator_response}")
        time.sleep(1)  # Simulate processing time

        # Update conversation for evaluator's turn
        current_conversation = "\n".join(conversation)

        # Evaluator's turn
        evaluator_response = get_response(evaluator_prompt, topic, essay, current_conversation)
        conversation.append(f"evaluator: {evaluator_response}")
        time.sleep(1)  # Simulate processing time

        print(f"Turn {turn + 1} completed")

        # Check if evaluator response includes "good"
        if "good" in evaluator_response.lower():
            feedback_good = True
            good_feedback = generator_response
            break

    # Save conversation to log file
    with open("conversations_log.txt", "w", encoding='utf-8') as f:
        for line in conversation:
            f.write(line + "\n")

    # Output the final feedback if it was marked as 'good'
    if feedback_good:
        print("Feedback process completed. Final feedback:")
        print(good_feedback)
    else:
        print("Max turns reached without 'good' feedback.")

# Collect user inputs for topic and essay
topic = input("Enter the IELTS essay topic: ")
essay = input("Enter the essay: ")

generate_feedback(topic, essay)