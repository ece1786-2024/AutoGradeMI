# Feedback Generator Prompt
import IELTS_rubrics as rubric
identity_prompt = (
    "You are an IELTS writing section examiner.\n"
)

task_prompt = """Given the following writing question and student essay, provide feedbacks for this student to improve the essay.
The feedback must be constructive, and provide suggestions for improvements.\n  
"""

def create_system_prompt(basic_rubric=True, criteria=True, band_score=True):
    prompt = identity_prompt
    
    if basic_rubric:
        prompt += f"Rubric: {rubric.BASIC_RUBRIC}\n\n"
    
    if criteria:
        prompt += f"Criteria: {rubric.CRITERIA}\n\n"

    if band_score:
        prompt += f"Band Scores (for each criteria): {rubric.BAND_SCORE}\n\n"

    prompt += task_prompt
    
    return prompt

def create_user_prompt(question, essay):
    msg = "Here is the writing question and the essay for feedback: \n"
    prompt = f"{msg}Question: {question}\n\nEssay: {essay}"
    return prompt

if __name__ == "__main__":
    print(" =========== USER PROMPT SAMPLE ============= ")
    print(create_user_prompt("Sample Question?", "Sample Essay"))
    print("\n\n =========== SYSTEM PROMPT SAMPLE ============= ")
    print(create_system_prompt())
