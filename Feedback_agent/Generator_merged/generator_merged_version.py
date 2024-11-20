from openai import OpenAI
import sys
import os

# Include the path to the rubric and samples
sys.path.append(os.path.abspath("rubric_and_sample"))
import IELTS_rubrics as rubric

# Initialize OpenAI client
client = OpenAI()

def generate_feedback(topic, essay, predicted_score, desired_score):
    # Generator prompt template
    generator_prompt = f"""
    You are an IELTS writing section examiner. Given the following writing question and student essay with student's score and student's desired score, provide comments and feedback for the student to improve the essay to reach the desired score.

    The feedback must be constructive and provide an additional section for overall suggestions and improvements and an overall summary.

    Try to think this step by step:
    1. How is the task response of the essay?
    2. How is the coherence and cohesion of the essay?
    3. How is the lexical resource of the essay?
    4. How is the grammatical range and accuracy of the essay?

    Writing Question: {topic}
    Student Essay: {essay}
    Student's Score: {predicted_score}
    Desired Score: {desired_score}

    Here is an IELTS Rubric for your reference:
    Rubric: {rubric}

    The feedback should address the following criteria:
    - Clarity: Feedback should be easy to understand and avoid ambiguous language.
    - Relevance: Feedback must address the main components of IELTS scoring.
    - Specificity: Feedback should point out exact issues and, if possible, provide examples.
    - Actionable Suggestions: Include constructive advice the writer can apply to improve.
    - Tone: Ensure feedback is encouraging and respectful.
    """
    
    # Call GPT-4 API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": generator_prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    # Get user input
    topic = input("Enter the writing question/topic: ")
    essay = input("Enter the student's essay: ")
    predicted_score = input("Enter the predicted score (e.g., 6.0): ")
    desired_score = input("Enter the desired score (e.g., 7.0): ")
    
    # Generate feedback
    print("\nGenerating feedback, please wait...")
    feedback = generate_feedback(topic, essay, predicted_score, desired_score)
    
    # Display feedback
    print("\n--- Feedback ---")
    print(feedback)

if __name__ == "__main__":
    main()
