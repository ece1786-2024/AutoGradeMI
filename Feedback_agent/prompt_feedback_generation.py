# Feedback Generator Prompt
import rubric_and_sample.IELTS_rubrics as rubric
import rubric_and_sample.feedback_sample as fs_sample
identity_prompt = (
    "You are an IELTS writing section examiner. Given the following writing question and student essay, provide comments and feedbacks for this student to improve on the essay.\n\n"
)

task_prompt = """
The feedback must be constructive, and provide an additional section for overall suggestions and improvements and an overall summary.\n  
"""

CoT_prompt = "Try to think this step by step. First, how is the task response of the essay? Second, how is the coherence and cohesion of the essay? Third, how is the lexical resource of the essay? Lastly, how is the grammatical range and accuracy of the essay? "

def create_system_prompt(basic_rubric=True, criteria=True, band_score=True, CoT=False, few_shot=False):
    prompt = identity_prompt
    
    if basic_rubric:
        prompt += f"Rubric: {rubric.BASIC_RUBRIC}\n\n"
    
    if criteria:
        prompt += f"Criteria: {rubric.CRITERIA}\n\n"

    if band_score:
        prompt += f"Band Scores (for each criteria): {rubric.BAND_SCORE}\n\n"

    if few_shot:
        msg = "Here is a sample of a feedback for another essay that can ONLY be used as a feedback example. Any content from this example MUST NOT affect your judgement on the student's essay."
        prompt += f"{msg}\nSample feedback: {fs_sample.few_shot_2}\n\n"

    if CoT:
        prompt += CoT_prompt

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
    print(create_system_prompt(CoT=True, few_shot=True))
