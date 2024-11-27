from openai import OpenAI
import time
import sys
import os
from datetime import datetime
from pathlib import Path

current_file = Path(__file__).resolve()
rubric_path = current_file.parent.parent
sys.path.append(str(rubric_path))

# Now you can import the IELTS_rubrics module
from rubric_and_sample import IELTS_rubrics as rubric

client = OpenAI()

def get_response(role_prompt, topic, essay, predicted_score, desired_score, conversation_history):
    try:
        prompt = role_prompt.format(topic=topic, essay=essay, predicted_score=predicted_score, desired_score=desired_score, rubric=rubric.BASIC_RUBRIC)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Based on this conversation, what would you say next?\n\n{conversation_history}"}
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return f"Error: {e}"

# Main function to process feedback until it is marked as 'good'
def generate_feedback(topic, essay, predicted_score, desired_score):
    # log_file_path = "conversations_log.txt"
    log_file_path = "conversations_log_progress_report.txt"

    if not topic.strip():
        raise ValueError("The topic is empty. Please provide a valid topic.")
    if not essay.strip():
        raise ValueError("The essay is empty. Please provide a valid essay.")
    
    # Define the generator and evaluator prompt templates
    generator_prompt = f"""
    You are an IELTS writing section examiner. Given the following writing question and student essay with student's score and student's desired score, provide comments and feedback for the student to improve the essay to reach the desired score.

    The feedback must be constructive and provide an additional section for overall suggestions and improvements and an overall summary.

    Try to think this step by step:
    1. How is the task response of the essay?
    2. How is the coherence and cohesion of the essay?
    3. How is the lexical resource of the essay?
    4. How is the grammatical range and accuracy of the essay?

    Writing Question: {topic}\n
    Student Essay: {essay}\n
    Student's Score: {predicted_score}\n
    Desired Score: {desired_score}\n

    Here is an IELTS Rubric for your reference:
    Rubric: {rubric}\n

    There will be an evaluation for the feedback provided. Please revise the feedback according to the evaluation if the feedback is not good enough.
    """

    evaluator_prompt = f"""
    You should evaluate the generated feedback for the IELTS writing response. 
    Writing Question: {topic}\n
    Student Essay: {essay}\n
    Student's Score: {predicted_score}\n
    Desired Score: {desired_score}\n

    Evaluate the provided feedback on the IELTS essay according to these criteria step by step: 
    Clarity: Ensure the feedback is easy to understand and avoids ambiguous language. Good feedback should use simple and precise wording, so the essay writer knows exactly what is meant.
    Relevance: Check if the feedback addresses the main components of IELTS scoring (Task Achievement, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy). Good feedback covers each of these areas as relevant to the essay.
    Specificity: Look for details in the feedback. Comments like 'Improve grammar' are not specific enough. Good feedback should mention exact issues (e.g., 'There are multiple subject-verb agreement errors') and, if possible, give examples from the essay.
    Actionable Suggestions: Verify that the feedback includes clear, constructive advice that the writer can apply to improve. Instead of saying 'The vocabulary is weak,' a good suggestion might be, 'Try to use more varied synonyms for common words like "important" (e.g., "crucial," "vital") to improve Lexical Resource.'
    Tone: Check if the feedback is encouraging and respectful. A supportive tone can help motivate the writer to make improvements.
    Continue to review the feedback until all criteria are met. If the feedback meets these criteria, respond with 'good.' If not, identify the areas needing improvement. You need to be strict and picky.
    """

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        conversation = [
            f"Topic: {topic}",
            f"Essay: {essay}",
            f"Predicted Score: {predicted_score}",
            f"Desired Score: {desired_score}",
            "Evaluator: Begin evaluating the generated feedback based on the rubric."
        ]
        feedback_good = False
        good_feedback = None
        repetitive_responses = set()

        for turn in range(10):
            current_conversation = "\n".join(conversation)

            # Generator's turn
            generator_response = get_response(generator_prompt, topic, essay, current_conversation)
            if generator_response in repetitive_responses:
                log_file.write("Repetitive responses detected. Exiting loop.\n")
                print("Repetitive responses detected. Exiting loop.")
                break
            repetitive_responses.add(generator_response)
            conversation.append(f"generator: {generator_response}")
            log_file.write(f"generator: {generator_response}\n")

            # Evaluator's turn
            current_conversation = "\n".join(conversation)
            evaluator_response = get_response(evaluator_prompt, topic, essay, current_conversation)
            conversation.append(f"evaluator: {evaluator_response}")
            log_file.write(f"evaluator: {evaluator_response}\n")

            # Check if evaluator deems feedback 'good'
            if "good" in evaluator_response.lower():
                feedback_good = True
                good_feedback = generator_response
                break

        # Log final result
        if feedback_good:
            #print("Feedback process completed successfully.")
            #print(f"Final feedback:\n{good_feedback}")
            return good_feedback
        else:
            #print("Failed to generate 'good' feedback within the turn limit. Please try again.")
            return "Failed to generate 'good' feedback within the turn limit. Please try again."


def main():
    # Collect user inputs for topic, essay, predicted score, and desired score
    topic = input("Enter the IELTS essay topic: ")
    essay = input("Enter the essay: ")
    predicted_score = input("Enter the predicted score: ")
    desired_score = input("Enter the desired score: ")

    # Call the generate_feedback function
    feedback = generate_feedback(topic, essay, predicted_score, desired_score)

    # Output the result
    print("Generated Feedback:")
    print(feedback)

if __name__ == "__main__":
    main()

#test topic: In some countries more and more people are becoming interested in finding out bout the history of the house or building they live in. What are the reasons for this? How can people research this?
#test essay: In recent years, many people are concerned to discover the history of their residence. This essay will explore what might be the causes and methods of how society can find out the information. The primary reason for exploring their residence is due to the huge investment in owning a property. Therefore, they are keen on taking precautious measures to secure their financing, which can be threatening if the property has terrible notorieties. For instance, if a criminal act has occurred in a building or a place, such as murder or others, people fear such incidents will somehow happen in the future. This bad reputation could drop the market value of the place. Besides that, homeowners are more concerned about the durability of their house structures due to climate change that affects every continent in the world. There are various methods to discover it. The first one is to find any news related to the districts you live in on the internet. Secondly, to interview local residents in the town as they have been living in a particular area for a long time, they must have heard many myths as well as experienced some by themselves. In this case, individuals can acquire first-hand evidence. Thirdly, paying a visit to the construction office as they can provide the details about materials to build the houses. In conclusion, in many countries, people are more intrigued by the history of the places they intend to live in, for various reasons such as their families' safety and money investment. There are many approaches to find out how the houses are built: internet, local persons and offices.
